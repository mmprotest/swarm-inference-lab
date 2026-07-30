"""Separate-process unsplit reference execution and split-stage validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.manifest import load_manifest, verify_manifest_shards
from swarm_inference.model.qwen3 import Qwen3Adapter
from swarm_inference.protocol.checksums import sha256_file


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    passed: bool
    distributed_token_ids: list[int]
    reference_token_ids: list[int]
    greedy_token_identity: bool
    intermediate_comparisons: list[dict[str, Any]]
    cache_replay_preserved_output: bool
    reference_phase: dict[str, Any]
    split_phase: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "distributed_token_ids": self.distributed_token_ids,
            "reference_token_ids": self.reference_token_ids,
            "greedy_token_identity": self.greedy_token_identity,
            "intermediate_comparisons": self.intermediate_comparisons,
            "cache_replay_preserved_output": self.cache_replay_preserved_output,
            "reference_phase": self.reference_phase,
            "split_phase": self.split_phase,
        }


def _torch_dtype(name: str) -> Any:
    import torch

    normalised = name.lower()
    mapping = {
        "f16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[normalised]
    except KeyError as exc:
        raise ValueError(f"unsupported validation dtype {name}") from exc


def execute_reference(
    *,
    model_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: str,
    dtype_name: str,
    stage_layer_ends: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    """Load the full model only inside the explicitly disclosed reference phase."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _torch_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_path, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)  # type: ignore[arg-type]
    model.eval()
    boundary_outputs: dict[int, Any] = {}
    hooks = []
    for layer_end in stage_layer_ends:
        layer_index = layer_end - 1

        def capture(
            module: Any,
            inputs: Any,
            output: Any,
            *,
            layer_end: int = layer_end,
        ) -> None:
            value = output[0] if isinstance(output, tuple) else output
            boundary_outputs[layer_end] = value.detach().float().cpu().numpy()

        hooks.append(model.model.layers[layer_index].register_forward_hook(capture))
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    generated: list[int] = []
    with torch.inference_mode():
        result = model(input_ids=input_ids, use_cache=True)
        logits = result.logits
        np.save(output_dir / "reference-final-logits.npy", logits.detach().float().cpu().numpy())
        for layer_end, value in boundary_outputs.items():
            np.save(output_dir / f"reference-layer-{layer_end:04d}.npy", value)
        next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_token)
        cache = result.past_key_values
        for _ in range(1, max_new_tokens):
            result = model(
                input_ids=torch.tensor([[generated[-1]]], device=device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = result.past_key_values
            generated.append(int(torch.argmax(result.logits[:, -1, :], dim=-1).item()))
    for hook in hooks:
        hook.remove()
    payload = {
        "token_ids": generated,
        "prompt_token_ids": input_ids.detach().cpu().reshape(-1).tolist(),
        "prompt_length": int(input_ids.shape[1]),
        "model_path": str(model_path),
        "device": device,
        "dtype": dtype_name,
        "full_model_loaded": True,
        "memory_counted_as_swarm": False,
        "reference_output_files": sorted(path.name for path in output_dir.glob("reference-*.npy")),
    }
    (output_dir / "reference.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def execute_reference_suite(
    *,
    model_id: str,
    model_revision: str,
    model_path: Path,
    requests_path: Path,
    device: str,
    dtype_name: str,
    stage_layer_ends: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    """Run all correctness-oracle requests in one isolated full-model process."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_root = output_dir / ".reference-boundaries"
    boundary_root.mkdir(parents=True, exist_ok=True)
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    if not isinstance(requests, list) or not requests:
        raise IntegrityError("reference suite input must be a non-empty request list")
    dtype = _torch_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        local_files_only=True,
    )
    process = psutil.Process()
    host_before = int(process.memory_info().rss)
    cuda_before = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        cuda_before = int(torch.cuda.memory_allocated(device))
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)  # type: ignore[arg-type]
    model.eval()
    model.requires_grad_(False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    model_load_s = time.perf_counter() - load_started
    boundary_outputs: dict[int, Any] = {}
    hooks = []
    for layer_end in stage_layer_ends:
        layer_index = layer_end - 1

        def capture(
            module: Any,
            inputs: Any,
            output: Any,
            *,
            layer_end: int = layer_end,
        ) -> None:
            value = output[0] if isinstance(output, tuple) else output
            boundary_outputs[layer_end] = value.detach().float().cpu().numpy()

        hooks.append(model.model.layers[layer_index].register_forward_hook(capture))
    configured_eos = getattr(model.config, "eos_token_id", None)
    eos_ids = (
        {int(value) for value in configured_eos}
        if isinstance(configured_eos, list)
        else ({int(configured_eos)} if configured_eos is not None else set())
    )
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for request in requests:
            request_id = str(request["request_id"])
            prompt_ids = [int(value) for value in request["prompt_token_ids"]]
            maximum = int(request["max_new_tokens"])
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            boundary_outputs.clear()
            prefill_started = time.perf_counter()
            forward = model(input_ids=input_ids, use_cache=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            prefill_s = time.perf_counter() - prefill_started
            request_boundary_dir = boundary_root / request_id
            request_boundary_dir.mkdir(parents=True, exist_ok=True)
            boundary_hashes: dict[str, str] = {}
            for layer_end, value in boundary_outputs.items():
                boundary_path = request_boundary_dir / f"reference-layer-{layer_end:04d}.npy"
                np.save(boundary_path, value)
                boundary_hashes[str(layer_end)] = sha256_file(boundary_path)
            generated: list[int] = []
            step_results: list[dict[str, Any]] = []
            cache = forward.past_key_values
            logits = forward.logits
            decode_s = 0.0
            for step in range(maximum):
                selected = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                top_values, top_indices = torch.topk(
                    logits[0, -1, :].float(),
                    k=min(10, int(logits.shape[-1])),
                )
                step_results.append(
                    {
                        "step": step,
                        "selected_token_id": selected,
                        "selected_token_text": tokenizer.decode([selected]),
                        "selected_token_logit": float(logits[0, -1, selected].float().item()),
                        "top_logits": [
                            {
                                "token_id": int(token_id),
                                "token_text": tokenizer.decode([int(token_id)]),
                                "logit": float(logit),
                            }
                            for token_id, logit in zip(
                                top_indices.tolist(),
                                top_values.tolist(),
                                strict=True,
                            )
                        ],
                    }
                )
                generated.append(selected)
                if selected in eos_ids or step + 1 >= maximum:
                    break
                decode_started = time.perf_counter()
                forward = model(
                    input_ids=torch.tensor([[selected]], device=device),
                    past_key_values=cache,
                    use_cache=True,
                )
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                decode_s += time.perf_counter() - decode_started
                cache = forward.past_key_values
                logits = forward.logits
            results.append(
                {
                    "request_id": request_id,
                    "name": request.get("name", request_id),
                    "prompt": request.get("prompt"),
                    "prompt_token_ids": prompt_ids,
                    "input_token_count": len(prompt_ids),
                    "generated_token_ids": generated,
                    "output_token_count": len(generated),
                    "decoded_text": tokenizer.decode(
                        generated,
                        skip_special_tokens=False,
                    ),
                    "steps": step_results,
                    "prefill_latency_s": prefill_s,
                    "decode_latency_s": decode_s,
                    "boundary_hashes": boundary_hashes,
                    "eos_token_ids": sorted(eos_ids),
                }
            )
    for hook in hooks:
        hook.remove()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    memory = {
        "host_rss_before_load": host_before,
        "host_rss_after_execution": int(process.memory_info().rss),
        "cuda_allocated_before_load": cuda_before,
        "cuda_allocated_after_execution": (
            int(torch.cuda.memory_allocated(device)) if device.startswith("cuda") else 0
        ),
        "cuda_peak_allocated": (
            int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
        ),
        "cuda_peak_reserved": (
            int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0
        ),
    }
    payload = {
        "phase": "independent-full-model-reference",
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_path": str(model_path),
        "device": device,
        "dtype": dtype_name,
        "full_model_loaded": True,
        "memory_counted_as_swarm": False,
        "model_load_s": model_load_s,
        "thinking_enabled": False,
        "greedy": True,
        "temperature": 0,
        "results": results,
        "memory": memory,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0) if device.startswith("cuda") else None),
        },
        "boundary_root": str(boundary_root),
    }
    (output_dir / "reference.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_reference_suite_subprocess(
    *,
    model_id: str,
    model_revision: str,
    model_path: Path,
    requests: list[dict[str, Any]],
    device: str,
    dtype_name: str,
    stage_layer_ends: list[int],
    output_dir: Path,
    log_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "reference-requests.json"
    requests_path.write_text(
        json.dumps(requests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "swarm_inference.model.reference",
        "--reference-suite",
        "--model-id",
        model_id,
        "--model-revision",
        model_revision,
        "--model-path",
        str(model_path),
        "--suite-input",
        str(requests_path),
        "--device",
        device,
        "--dtype",
        dtype_name,
        "--stage-layer-ends",
        ",".join(str(value) for value in stage_layer_ends),
        "--output-dir",
        str(output_dir),
    ]
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    source_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            result.stdout + ("\n" if result.stdout else "") + result.stderr,
            encoding="utf-8",
        )
    if result.returncode != 0:
        raise IntegrityError(
            "reference suite subprocess failed: "
            f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return cast(
        dict[str, Any],
        json.loads((output_dir / "reference.json").read_text(encoding="utf-8")),
    )


def run_reference_subprocess(
    *,
    model_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: str,
    dtype_name: str,
    stage_layer_ends: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "swarm_inference.model.reference",
        "--reference-only",
        "--model-path",
        str(model_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--dtype",
        dtype_name,
        "--stage-layer-ends",
        ",".join(str(value) for value in stage_layer_ends),
        "--output-dir",
        str(output_dir),
    ]
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    source_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if result.returncode != 0:
        raise IntegrityError(
            "unsplit reference subprocess failed: "
            f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return cast(
        dict[str, Any],
        json.loads((output_dir / "reference.json").read_text(encoding="utf-8")),
    )


def validate_qwen_correctness(
    *,
    shard_root: str | Path,
    model_path: str | Path,
    prompt: str,
    max_new_tokens: int = 4,
    device: str = "cpu",
    dtype_name: str = "float32",
    atol: float = 1e-5,
    rtol: float = 1e-4,
    output_dir: str | Path,
    replay_stage_id: int = 0,
) -> CorrectnessResult:
    import torch
    from transformers import AutoConfig

    root = Path(shard_root).expanduser().resolve()
    source = Path(model_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root / "manifest.json")
    verify_manifest_shards(manifest, root)
    reference = run_reference_subprocess(
        model_path=source,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype_name=dtype_name,
        stage_layer_ends=[stage.layer_end for stage in manifest.stages],
        output_dir=output,
    )
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    adapter = Qwen3Adapter()
    if not adapter.supports(config):
        raise IntegrityError("validation source is not supported dense Qwen3")
    dtype = _torch_dtype(dtype_name)
    modules = []
    loaded_proofs = []
    for stage in manifest.stages:
        module = adapter.create_stage_module(
            config,
            stage,
            torch.device(device),
            dtype,
        )
        loaded = adapter.load_stage_weights(
            module,
            root / f"stage-{stage.stage_id:03d}",
            manifest=manifest,
        )
        modules.append(module)
        loaded_proofs.append(
            {
                "stage_id": stage.stage_id,
                "loaded_tensor_names": loaded,
                "logical_loaded_bytes": stage.required_memory_bytes,
            }
        )
    prompt_ids = torch.tensor(
        [reference["prompt_token_ids"]],
        dtype=torch.long,
        device=device,
    )
    stage_inputs: dict[int, Any] = {}
    comparisons: list[dict[str, Any]] = []
    with torch.inference_mode():
        current: Any = prompt_ids
        for index, module in enumerate(modules):
            stage_inputs[index] = current.detach().clone()
            current = module.forward(
                current,
                request_id="split-validation",
                token_position=0,
                cache_generation=0,
                use_cache=True,
            )
            if index < len(modules) - 1:
                reference_boundary = np.load(
                    output / f"reference-layer-{manifest.stages[index].layer_end:04d}.npy",
                    allow_pickle=False,
                )
            else:
                reference_boundary = np.load(
                    output / "reference-final-logits.npy",
                    allow_pickle=False,
                )
            distributed_boundary = current.detach().float().cpu().numpy()
            close = bool(
                np.allclose(
                    distributed_boundary,
                    reference_boundary,
                    atol=atol,
                    rtol=rtol,
                )
            )
            max_error = float(
                np.max(
                    np.abs(
                        distributed_boundary.astype(np.float64)
                        - reference_boundary.astype(np.float64)
                    )
                )
            )
            comparisons.append(
                {
                    "stage_id": manifest.stages[index].stage_id,
                    "layer_end": manifest.stages[index].layer_end,
                    "within_tolerance": close,
                    "maximum_absolute_error": max_error,
                    "atol": atol,
                    "rtol": rtol,
                }
            )
        generated = [int(torch.argmax(current[:, -1, :], dim=-1).item())]

        # Reconstruct one stage from the exact prefill stage input before decode.
        if not 0 <= replay_stage_id < len(modules):
            raise ValueError("replay_stage_id is outside the stage range")
        replacement = adapter.create_stage_module(
            config,
            manifest.stages[replay_stage_id],
            torch.device(device),
            dtype,
        )
        adapter.load_stage_weights(
            replacement,
            root / f"stage-{replay_stage_id:03d}",
            manifest=manifest,
        )
        replacement.forward(
            stage_inputs[replay_stage_id],
            request_id="split-validation",
            token_position=0,
            cache_generation=0,
            use_cache=True,
        )
        modules[replay_stage_id] = replacement

        prompt_length = int(reference["prompt_length"])
        for generation_index in range(1, max_new_tokens):
            current = torch.tensor(
                [[generated[-1]]],
                dtype=torch.long,
                device=device,
            )
            absolute_position = prompt_length + generation_index - 1
            for module in modules:
                current = module.forward(
                    current,
                    request_id="split-validation",
                    token_position=absolute_position,
                    cache_generation=0,
                    use_cache=True,
                )
            generated.append(int(torch.argmax(current[:, -1, :], dim=-1).item()))
    identity = generated == [int(value) for value in reference["token_ids"]]
    intermediate_ok = all(item["within_tolerance"] for item in comparisons)
    replay_ok = identity
    split_phase = {
        "device": device,
        "dtype": dtype_name,
        "coordinator_full_model_loaded": False,
        "validation_process_holds_multiple_stage_modules": True,
        "memory_counted_as_swarm": False,
        "stage_load_proofs": loaded_proofs,
        "replayed_stage_id": replay_stage_id,
    }
    result = CorrectnessResult(
        passed=identity and intermediate_ok and replay_ok,
        distributed_token_ids=generated,
        reference_token_ids=[int(value) for value in reference["token_ids"]],
        greedy_token_identity=identity,
        intermediate_comparisons=comparisons,
        cache_replay_preserved_output=replay_ok,
        reference_phase=reference,
        split_phase=split_phase,
    )
    (output / "correctness.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-suite", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--suite-input")
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--stage-layer-ends", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    stage_ends = [int(value) for value in arguments.stage_layer_ends.split(",") if value]
    if arguments.reference_suite:
        if (
            arguments.suite_input is None
            or arguments.model_id is None
            or arguments.model_revision is None
        ):
            parser.error(
                "--reference-suite requires --suite-input, --model-id, and --model-revision"
            )
        execute_reference_suite(
            model_id=arguments.model_id,
            model_revision=arguments.model_revision,
            model_path=Path(arguments.model_path).resolve(),
            requests_path=Path(arguments.suite_input).resolve(),
            device=arguments.device,
            dtype_name=arguments.dtype,
            stage_layer_ends=stage_ends,
            output_dir=Path(arguments.output_dir).resolve(),
        )
    elif arguments.reference_only:
        if arguments.prompt is None or arguments.max_new_tokens is None:
            parser.error("--reference-only requires --prompt and --max-new-tokens")
        execute_reference(
            model_path=Path(arguments.model_path).resolve(),
            prompt=arguments.prompt,
            max_new_tokens=arguments.max_new_tokens,
            device=arguments.device,
            dtype_name=arguments.dtype,
            stage_layer_ends=stage_ends,
            output_dir=Path(arguments.output_dir).resolve(),
        )
    else:
        parser.error("select --reference-only or --reference-suite")


if __name__ == "__main__":
    main()
