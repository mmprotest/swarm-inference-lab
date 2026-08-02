"""Real OLMoE activation capture and isolated expert RPC probe.

This path deliberately does not claim to be the Colibri expert-redirection
hook.  It captures a genuine routed activation with the source Hugging Face
model, releases that model, and then executes the selected source experts in
independent worker processes through the backend-neutral RPC ABI.
"""

from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from swarm_inference.experiments.experiment_010.coordinator import (
    StableExpertCoordinator,
    compare_layer_results,
)
from swarm_inference.experiments.experiment_010.expert import (
    execute_expert,
    reduce_partials,
    safetensors_expert_loader,
    safetensors_expert_ownership_entry,
)
from swarm_inference.experiments.experiment_010.schemas import (
    DeterminismMode,
    EvidenceCategory,
    ReductionMode,
    WorkerBudget,
    WorkerManifest,
)
from swarm_inference.experiments.experiment_010.transport import ExpertTransportClient
from swarm_inference.experiments.experiment_010.verification import (
    reconcile_expert_ownership,
)
from swarm_inference.experiments.experiment_010.worker import ExpertWorkerManager
from swarm_inference.protocol.checksums import sha256_bytes


@dataclass(frozen=True, slots=True)
class LevelAActivationCapture:
    activation: np.ndarray
    expert_ids: tuple[int, ...]
    routing_weights: tuple[float, ...]
    evidence: dict[str, Any]


def _model_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        path = root / name
        if path.is_file():
            digest.update(name.encode("ascii"))
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def capture_level_a_activation(
    model_path: Path,
    *,
    prompt: str = "The future of distributed inference depends on",
    layer_id: int = 0,
    device: str = "cuda:0",
) -> LevelAActivationCapture:
    """Capture one real last-token activation and its top-k OLMoE route."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = model_path.expanduser().resolve()
    if not (root / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("Level A source model needs model.safetensors.index.json")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Level A CUDA activation capture requested without CUDA")
    started = time.perf_counter_ns()
    capture: dict[str, Any] = {}
    model: Any | None = None
    tokenizer: Any | None = None
    output: Any | None = None
    encoded: Any | None = None
    input_ids: Any | None = None
    attention_mask: Any | None = None
    handles: list[Any] = []
    allocated_before = int(torch.cuda.memory_allocated()) if device.startswith("cuda") else 0
    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()
        block = model.model.layers[layer_id].mlp

        def capture_input(_module: Any, arguments: tuple[Any, ...]) -> None:
            capture["hidden_states"] = arguments[0].detach().to(device="cpu", dtype=torch.float32)

        def capture_output(
            _module: Any, _arguments: tuple[Any, ...], output: tuple[Any, Any]
        ) -> None:
            capture["router_logits"] = output[1].detach().to(device="cpu", dtype=torch.float32)

        handles.append(block.register_forward_pre_hook(capture_input))
        handles.append(block.register_forward_hook(capture_output))
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        if "hidden_states" not in capture or "router_logits" not in capture:
            raise RuntimeError("OLMoE layer hooks did not observe activation and router logits")
        hidden = capture["hidden_states"].reshape(-1, capture["hidden_states"].shape[-1])
        router_logits = capture["router_logits"].reshape(-1, capture["router_logits"].shape[-1])
        last_activation = hidden[-1:].numpy().copy()
        probabilities = torch.softmax(router_logits[-1], dim=-1, dtype=torch.float32)
        top_k = int(block.top_k)
        routing_weights, selected_experts = torch.topk(probabilities, top_k)
        if bool(block.norm_topk_prob):
            routing_weights /= routing_weights.sum()
        input_token_ids = input_ids[0].detach().cpu().tolist()
        next_token_id = int(torch.argmax(output.logits[0, -1]).item())
        peak_allocated = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        evidence = {
            "status": "COMPLETED",
            "category": EvidenceCategory.MEASURED_SINGLE_HOST.value,
            "model_path": str(root),
            "model_fingerprint": _model_fingerprint(root),
            "layer_id": layer_id,
            "prompt": prompt,
            "input_token_ids": input_token_ids,
            "captured_token_position": len(input_token_ids) - 1,
            "activation_shape": list(last_activation.shape),
            "activation_dtype_at_model_boundary": "bfloat16",
            "transport_reference_dtype": "float32",
            "activation_sha256": sha256_bytes(last_activation.tobytes()),
            "expert_ids": [int(item) for item in selected_experts.tolist()],
            "routing_weights": [float(item) for item in routing_weights.tolist()],
            "router_logits_sha256": sha256_bytes(router_logits[-1].numpy().tobytes()),
            "next_token_id_before_remote_probe": next_token_id,
            "gpu_allocated_before_bytes": allocated_before,
            "gpu_peak_allocated_bytes": peak_allocated,
            "capture_elapsed_ns": time.perf_counter_ns() - started,
            "generation_continued_with_remote_result": False,
            "colibri_expert_hook_used": False,
        }
        return LevelAActivationCapture(
            activation=np.ascontiguousarray(last_activation, dtype=np.float32),
            expert_ids=tuple(int(item) for item in selected_experts.tolist()),
            routing_weights=tuple(float(item) for item in routing_weights.tolist()),
            evidence=evidence,
        )
    finally:
        for handle in handles:
            handle.remove()
        capture.clear()
        output = None
        encoded = None
        input_ids = None
        attention_mask = None
        tokenizer = None
        model = None
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _worker_budget(
    worker_id: str,
    root: Path,
    resident_bytes: int,
    cpu: int,
) -> WorkerBudget:
    return WorkerBudget(
        worker_id=worker_id,
        memory_budget_bytes=resident_bytes,
        expert_residency_budget_bytes=resident_bytes,
        cache_budget_bytes=resident_bytes,
        thread_count=1,
        cpu_affinity=[cpu],
        storage_directory=str(root),
        device="cpu",
        backend="numpy-source-bfloat16-to-fp32",
        physical_memory_limit=False,
    )


def execute_level_a_expert_rpc(
    capture: LevelAActivationCapture,
    *,
    model_path: Path,
    root: Path,
    repeats: int,
) -> dict[str, Any]:
    """Execute captured routed experts on two disjoint source-weight workers."""

    source = model_path.expanduser().resolve()
    layer_id = int(capture.evidence["layer_id"])
    entries = {
        expert_id: safetensors_expert_ownership_entry(
            source, layer_id=layer_id, expert_id=expert_id
        )
        for expert_id in capture.expert_ids
    }
    worker_ids = ("level-a-source-0", "level-a-source-1")
    assignments: dict[str, list[int]] = {worker_id: [] for worker_id in worker_ids}
    for index, expert_id in enumerate(capture.expert_ids):
        assignments[worker_ids[index % len(worker_ids)]].append(expert_id)
    loader = safetensors_expert_loader(source)
    weights_by_expert = dict(zip(capture.expert_ids, capture.routing_weights, strict=True))
    reference_partials = []
    for worker_id in sorted(worker_ids):
        partial = np.zeros_like(capture.activation, dtype=np.float32)
        for expert_id in assignments[worker_id]:
            weights = loader(layer_id, expert_id)
            partial += np.float32(weights_by_expert[expert_id]) * execute_expert(
                capture.activation, weights
            )
            del weights
        reference_partials.append((worker_id, partial))
    reference = reduce_partials(reference_partials, mode=ReductionMode.FIXED_ORDER_FP32)
    del loader, reference_partials
    gc.collect()

    manager = ExpertWorkerManager(root / "workers")
    clients: dict[str, ExpertTransportClient] = {}
    processes = {}
    manifests = []
    universal_workers = []
    budgets = []
    ownership = {}
    affinity = psutil.Process().cpu_affinity()
    try:
        for index, worker_id in enumerate(worker_ids):
            owned_entries = [entries[expert_id] for expert_id in assignments[worker_id]]
            resident_bytes = sum(int(item["logical_bytes"]) for item in owned_entries)
            budget = _worker_budget(
                worker_id,
                root / worker_id,
                resident_bytes,
                affinity[index % len(affinity)],
            )
            process = manager.start(
                worker_id=worker_id,
                model_id="allenai/OLMoE-1B-7B-0125-Instruct",
                model_revision=str(capture.evidence["model_fingerprint"]),
                quantization_fingerprint="source-bfloat16/transport-fp32",
                model_fingerprint=str(capture.evidence["model_fingerprint"]),
                owned_experts=owned_entries,
                budget=budget,
                loader_type="safetensors",
                model_path=source,
            )
            client = ExpertTransportClient(process.endpoint)
            clients[worker_id] = client
            processes[worker_id] = process
            universal_workers.append(
                {
                    "worker_id": process.worker_id,
                    "process_id": process.process.pid,
                    "control_endpoint": process.control_endpoint,
                    "data_endpoint": process.endpoint,
                    "negotiated_protocol": process.negotiated_protocol,
                    "identity": process.universal_identity,
                    "capabilities": process.universal_capabilities,
                    "initial_heartbeat": process.initial_heartbeat,
                    "lifecycle_owner": ("ExpertWorkerManager via UniversalWorkerClient"),
                }
            )
            budgets.append(budget.model_dump(mode="json"))
            for expert_id in assignments[worker_id]:
                ownership[(layer_id, expert_id)] = worker_id
        coordinator = StableExpertCoordinator(
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_revision=str(capture.evidence["model_fingerprint"]),
            quantization_fingerprint="source-bfloat16/transport-fp32",
            latent_dimension=int(capture.activation.shape[1]),
            clients=clients,
            whole_ownership=ownership,
        )
        rows = []
        replay_outputs = []
        for repeat in range(repeats):
            result = coordinator.execute_whole_layer(
                capture.activation,
                layer_id=layer_id,
                expert_ids=list(capture.expert_ids),
                routing_weights=list(capture.routing_weights),
                coalesced=True,
                determinism=DeterminismMode.QUALITY_BOUNDED,
                request_id=f"level-a-real-activation-{repeat}",
                timeout_s=120,
            )
            comparison = compare_layer_results(reference, result.output)
            replay_outputs.append(result.output.copy())
            rows.append(
                {
                    "configuration": "level_a_real_activation_direct_tcp",
                    "repeat": repeat,
                    "throughput": 1e9 / result.metrics["total_ns"],
                    "latency_ms": result.metrics["total_ns"] / 1e6,
                    "p95_latency_ms": result.metrics["total_ns"] / 1e6,
                    "ttft_ms": None,
                    "token_identity": None,
                    "correctness_mode": "quality_bounded",
                    "messages_per_layer": result.metrics["messages_per_layer"],
                    "activation_payload_bytes": result.metrics["activation_payload_bytes"],
                    "selected_experts": len(capture.expert_ids),
                    "category": EvidenceCategory.MEASURED_SINGLE_HOST.value,
                    "accounting_mode": "capacity-isolation operator probe",
                    **comparison,
                }
            )
        for worker_id in worker_ids:
            manifests.append(
                WorkerManifest.model_validate(clients[worker_id].control("manifest")["manifest"])
            )
        reconciliation = reconcile_expert_ownership(
            manifests,
            expected={(layer_id, expert_id) for expert_id in capture.expert_ids},
        )
        return {
            "status": "COMPLETED",
            "category": EvidenceCategory.MEASURED_SINGLE_HOST.value,
            "rows": rows,
            "manifests": [item.model_dump(mode="json") for item in manifests],
            "universal_workers": universal_workers,
            "process_records": manager.lifecycle_records,
            "budgets": budgets,
            "ownership": reconciliation,
            "source_expert_inventory": list(entries.values()),
            "reference_sha256": sha256_bytes(reference.tobytes()),
            "exact_operator_equivalence": all(row["exact"] for row in rows),
            "remote_fixed_replay_exact": (
                all(np.array_equal(replay_outputs[0], item) for item in replay_outputs[1:])
                if len(replay_outputs) >= 2
                else None
            ),
            "maximum_relative_l2_error": max(float(row["relative_l2_error"]) for row in rows),
            "real_model_activation": True,
            "real_model_weights": True,
            "independent_worker_processes": len({item.process_id for item in manifests})
            == len(manifests),
            "generation_continued": False,
            "colibri_expert_hook_used": False,
            "coordinator_weight_state_at_rpc": "reference arrays released before workers started",
            "source_weight_reencoding": "bfloat16 source tensors converted to FP32 in worker",
        }
    finally:
        manager.close()
