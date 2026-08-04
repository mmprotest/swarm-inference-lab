"""Authoritative local reference, layer profiling, and bytewise exactness checks."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarm_inference.experiments.experiment_011.partition import StagePlan
from swarm_inference.experiments.experiment_011.tensor_transport import tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class LocalReferenceResult:
    prompt: str
    prompt_id: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    elapsed_seconds: float
    throughput_tps: float
    ttft_seconds: float
    inter_token_latencies_seconds: tuple[float, ...]
    layer_execution_ns: tuple[int, ...]
    capture_directory: str
    model_revision: str
    tokenizer_revision: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "prompt_token_ids",
            "generated_token_ids",
            "inter_token_latencies_seconds",
            "layer_execution_ns",
        ):
            value[name] = list(value[name])
        return value


def _write_tensor(path: Path, name: str, tensor: torch.Tensor) -> dict[str, Any]:
    raw = tensor_raw_bytes(tensor)
    target = path / f"{name}.bin"
    target.write_bytes(raw)
    return {
        "name": name,
        "path": str(target),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run_local_reference(
    *,
    model_path: Path,
    workload_reference_path: Path,
    plan: StagePlan,
    generated_token_count: int,
    output_directory: Path,
) -> LocalReferenceResult:
    workload = json.loads(workload_reference_path.read_text(encoding="utf-8"))
    prompt = str(workload["prompt"])
    prompt_id = str(workload["prompt_id"])
    expected_prompt_ids = [int(value) for value in workload["prompt_ids"]]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenized = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenized != expected_prompt_ids:
        raise ValueError("current tokenizer does not reproduce Experiment 010 input token IDs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    model.eval()
    layer_outputs: dict[int, torch.Tensor] = {}
    layer_events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    layer_samples: list[list[int]] = [[] for _ in model.model.layers]
    hooks = []

    def make_pre_hook(layer_id: int):
        def pre_hook(_: Any, __: Any) -> None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            layer_events[layer_id] = (start, end)

        return pre_hook

    def make_post_hook(layer_id: int):
        def post_hook(_: Any, __: Any, output: Any) -> None:
            layer_events[layer_id][1].record()
            layer_outputs[layer_id] = output[0].detach()

        return post_hook

    for layer_id, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_pre_hook(layer_id)))
        hooks.append(layer.register_forward_hook(make_post_hook(layer_id)))
    capture_root = output_directory.resolve()
    capture_root.mkdir(parents=True, exist_ok=True)
    generated: list[int] = []
    completion_times: list[int] = []
    past_key_values = None
    current_ids = torch.tensor([expected_prompt_ids], dtype=torch.long, device="cuda:0")
    measurement_started = time.perf_counter_ns()
    with torch.inference_mode():
        for token_position in range(generated_token_count):
            layer_outputs.clear()
            layer_events.clear()
            forward_started = time.perf_counter_ns()
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_router_logits=True,
                return_dict=True,
            )
            torch.cuda.synchronize(0)
            _ = time.perf_counter_ns() - forward_started
            for layer_id, (start_event, end_event) in layer_events.items():
                layer_samples[layer_id].append(int(start_event.elapsed_time(end_event) * 1e6))
            logits = outputs.logits[:, -1, :]
            token = int(torch.argmax(logits, dim=-1).item())
            generated.append(token)
            completion_times.append(time.perf_counter_ns())
            token_root = capture_root / f"token-{token_position:04d}"
            for assignment in plan.assignments:
                stage_root = token_root / f"stage-{assignment.stage_id}"
                stage_root.mkdir(parents=True, exist_ok=True)
                manifest = [
                    _write_tensor(
                        stage_root,
                        "stage_boundary_hidden",
                        layer_outputs[assignment.layer_end - 1],
                    )
                ]
                for global_layer_id in assignment.layer_ids:
                    manifest.append(
                        _write_tensor(
                            stage_root,
                            f"router_layer_{global_layer_id:02d}_fp32",
                            outputs.router_logits[global_layer_id].float(),
                        )
                    )
                if assignment.owns_final_norm:
                    manifest.append(
                        _write_tensor(stage_root, "final_hidden", outputs.hidden_states[-1])
                    )
                    manifest.append(
                        _write_tensor(
                            stage_root,
                            "pre_sampling_logits_fp32",
                            outputs.logits[:, -1, :].float(),
                        )
                    )
                (stage_root / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            past_key_values = outputs.past_key_values
            current_ids = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
    measurement_ended = time.perf_counter_ns()
    for hook in hooks:
        hook.remove()
    elapsed = (measurement_ended - measurement_started) / 1e9
    ttft = (completion_times[0] - measurement_started) / 1e9
    itls = tuple((right - left) / 1e9 for left, right in itertools.pairwise(completion_times))
    layer_execution_ns = tuple(
        int(statistics.median(samples)) if samples else 0 for samples in layer_samples
    )
    result = LocalReferenceResult(
        prompt=prompt,
        prompt_id=prompt_id,
        prompt_token_ids=tuple(expected_prompt_ids),
        generated_token_ids=tuple(generated),
        elapsed_seconds=elapsed,
        throughput_tps=generated_token_count / elapsed,
        ttft_seconds=ttft,
        inter_token_latencies_seconds=itls,
        layer_execution_ns=layer_execution_ns,
        capture_directory=str(capture_root),
        model_revision=plan.model_revision,
        tokenizer_revision=plan.tokenizer_revision,
    )
    (output_directory / "reference_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del outputs, past_key_values, current_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _tensor_from_bytes(path: Path, *, dtype: str, shape: list[int]) -> torch.Tensor:
    payload = path.read_bytes()
    byte_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
        "uint8": torch.uint8,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported captured dtype {dtype!r}")
    return byte_tensor.view(dtype_map[dtype]).reshape(shape)


def compare_capture_trees(
    *,
    local_capture_directory: Path,
    distributed_capture_directory: Path,
    session_id: str,
    prompt_id: str,
    reproduction_command: str,
) -> dict[str, Any]:
    local_manifests = sorted(local_capture_directory.glob("token-*/stage-*/manifest.json"))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for local_manifest_path in local_manifests:
        relative = local_manifest_path.relative_to(local_capture_directory)
        distributed_manifest_path = distributed_capture_directory / session_id / relative
        if not distributed_manifest_path.is_file():
            missing.append(str(distributed_manifest_path))
            continue
        local_entries = {
            row["name"]: row for row in json.loads(local_manifest_path.read_text(encoding="utf-8"))
        }
        distributed_entries = {
            row["name"]: row
            for row in json.loads(distributed_manifest_path.read_text(encoding="utf-8"))
        }
        token_position = int(relative.parts[0].split("-")[-1])
        stage_id = int(relative.parts[1].split("-")[-1])
        for name, local in local_entries.items():
            distributed = distributed_entries.get(name)
            if distributed is None:
                missing.append(f"{distributed_manifest_path}:{name}")
                continue
            local_path = Path(local["path"])
            distributed_path = Path(distributed["path"])
            local_raw = local_path.read_bytes()
            distributed_raw = distributed_path.read_bytes()
            byte_match = local_raw == distributed_raw
            local_tensor = _tensor_from_bytes(
                local_path, dtype=str(local["dtype"]), shape=list(local["shape"])
            ).float()
            distributed_tensor = _tensor_from_bytes(
                distributed_path,
                dtype=str(distributed["dtype"]),
                shape=list(distributed["shape"]),
            ).float()
            if local_tensor.shape != distributed_tensor.shape:
                max_absolute = float("inf")
                relative_l2 = float("inf")
            else:
                difference = local_tensor - distributed_tensor
                max_absolute = float(difference.abs().max().item())
                denominator = float(torch.linalg.vector_norm(local_tensor).item())
                numerator = float(torch.linalg.vector_norm(difference).item())
                relative_l2 = numerator / denominator if denominator else numerator
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "token_position": token_position,
                    "stage_id": stage_id,
                    "layer_boundary": name,
                    "local_tensor_path": str(local_path),
                    "distributed_tensor_path": str(distributed_path),
                    "local_sha256": hashlib.sha256(local_raw).hexdigest(),
                    "distributed_sha256": hashlib.sha256(distributed_raw).hexdigest(),
                    "byte_match": byte_match,
                    "maximum_absolute_difference_fp32": max_absolute,
                    "relative_l2_error_fp32": relative_l2,
                    "reproduction_command": reproduction_command,
                }
            )
    mismatches = [
        row
        for row in rows
        if not row["byte_match"]
        or row["maximum_absolute_difference_fp32"] != 0.0
        or row["relative_l2_error_fp32"] != 0.0
    ]
    return {
        "prompt_id": prompt_id,
        "comparison_count": len(rows),
        "missing": missing,
        "mismatch_count": len(mismatches),
        "byte_mismatch_count": sum(not row["byte_match"] for row in rows),
        "maximum_absolute_difference_fp32": max(
            (float(row["maximum_absolute_difference_fp32"]) for row in rows), default=0.0
        ),
        "maximum_relative_l2_error_fp32": max(
            (float(row["relative_l2_error_fp32"]) for row in rows), default=0.0
        ),
        "rows": rows,
        "mismatches": mismatches,
        "exact": not missing and not mismatches,
    }
