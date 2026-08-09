"""Independent full-model reference process for Experiment 012 H012-012."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_reference(
    *,
    model_path: Path,
    model_id: str,
    revision: str,
    prompt: str,
    max_new_tokens: int,
    output_directory: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("H012-012 independent reference requires CUDA")
    torch.manual_seed(12)
    torch.cuda.manual_seed_all(12)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    load_started_ns = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    torch.cuda.synchronize(device)
    load_elapsed_ns = time.perf_counter_ns() - load_started_ns

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    sequence = prompt_ids.clone()
    sequence_mask = attention_mask.clone()
    steps: list[dict[str, Any]] = []
    manual_ids: list[int] = []
    inference_started_ns = time.perf_counter_ns()
    with torch.inference_mode():
        for step_index in range(max_new_tokens):
            step_started_ns = time.perf_counter_ns()
            output = model(
                input_ids=sequence,
                attention_mask=sequence_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            torch.cuda.synchronize(device)
            hidden_device = output.hidden_states[-1][0, -1, :]
            logits_device = output.logits[0, -1, :]
            projected_device = model.lm_head(hidden_device)
            hidden = hidden_device.float().contiguous().cpu()
            logits = logits_device.float().contiguous().cpu()
            projected = projected_device.float().contiguous().cpu()
            projection_max_abs_error = float((logits - projected).abs().max().item())
            token_id = int(torch.argmax(logits).item())
            hidden_bytes = hidden.numpy().astype("<f4", copy=False).tobytes()
            logits_bytes = logits.numpy().astype("<f4", copy=False).tobytes()
            hidden_path = output_directory / f"step-{step_index:02d}.hidden.f32"
            logits_path = output_directory / f"step-{step_index:02d}.logits.f32"
            _write_bytes(hidden_path, hidden_bytes)
            _write_bytes(logits_path, logits_bytes)
            manual_ids.append(token_id)
            steps.append(
                {
                    "step_index": step_index,
                    "sequence_length": int(sequence.shape[1]),
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                    "score": float(logits[token_id].item()),
                    "hidden_shape": list(hidden.shape),
                    "hidden_dtype": "float32-le",
                    "hidden_path": str(hidden_path.resolve()),
                    "hidden_sha256": _sha256_bytes(hidden_bytes),
                    "logits_shape": list(logits.shape),
                    "logits_dtype": "float32-le",
                    "logits_path": str(logits_path.resolve()),
                    "logits_sha256": _sha256_bytes(logits_bytes),
                    "finite": bool(torch.isfinite(hidden).all() and torch.isfinite(logits).all()),
                    "projection_max_abs_error": projection_max_abs_error,
                    "elapsed_ns": time.perf_counter_ns() - step_started_ns,
                }
            )
            next_token = torch.tensor([[token_id]], dtype=sequence.dtype, device=device)
            sequence = torch.cat((sequence, next_token), dim=1)
            sequence_mask = torch.cat(
                (sequence_mask, torch.ones_like(next_token, dtype=sequence_mask.dtype)), dim=1
            )
        torch.cuda.synchronize(device)
        manual_elapsed_ns = time.perf_counter_ns() - inference_started_ns
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize(device)
    generated_ids = [int(value) for value in generated[0, prompt_ids.shape[1] :].tolist()]
    result = {
        "schema_version": "1.0",
        "evidence_class": "real_immutable_full_model_reference",
        "process_id": os.getpid(),
        "full_model_loaded": True,
        "memory_counted_as_swarm": False,
        "model_id": model_id,
        "model_revision": revision,
        "model_path": str(model_path.resolve()),
        "model_class": type(model).__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "prompt": prompt,
        "prompt_token_ids": [int(value) for value in prompt_ids[0].tolist()],
        "max_new_tokens": max_new_tokens,
        "manual_token_ids": manual_ids,
        "generate_token_ids": generated_ids,
        "manual_matches_generate": manual_ids == generated_ids,
        "decoded_new_tokens": tokenizer.decode(manual_ids, skip_special_tokens=False),
        "steps": steps,
        "load_elapsed_ns": load_elapsed_ns,
        "manual_inference_elapsed_ns": manual_elapsed_ns,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_index": device.index,
            "model_dtype": str(next(model.parameters()).dtype),
        },
    }
    if not result["manual_matches_generate"]:
        raise RuntimeError("manual full-model argmax loop differs from ordinary generate")
    if not all(step["finite"] for step in steps):
        raise RuntimeError("reference produced a non-finite hidden state or logit")
    _write_json(output_directory / "reference.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run H012-012 full-model reference")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_reference(
        model_path=args.model_path.resolve(),
        model_id=args.model_id,
        revision=args.revision,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        output_directory=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
