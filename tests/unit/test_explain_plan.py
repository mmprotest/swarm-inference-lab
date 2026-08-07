from __future__ import annotations

from swarm_inference.cluster.orchestrator import ClusterRunSummary
from swarm_inference.commands.run import _plan_explanation
from swarm_inference.coordinator.canonical_planner import CanonicalPlanningDecision
from swarm_inference.engines.interfaces import (
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionPlan,
    PhasePlan,
)


def _plan() -> ExecutionPlan:
    roles = {
        "node-a/worker": "critical_path_stage",
        "node-b/worker": "tensor_rpc_compute",
    }
    return ExecutionPlan(
        plan_id="plan-qwen3-moe",
        engine_id="llamacpp-rpc",
        model_fingerprint="sha256:" + "b" * 64,
        execution_identity="sha256:" + "c" * 64,
        objective="capacity",
        topology="llamacpp-rpc-2-host",
        worker_roles=roles,
        idle_workers={"node-c/worker": "weak node adds no required capacity"},
        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
        predicted_ttft_ms=100,
        predicted_decode_tokens_s=2,
        predicted_aggregate_tokens_s=2,
        predicted_network_bytes=None,
        predicted_messages_per_token=None,
        predicted_bytes_per_token=None,
        predicted_serial_waits_per_token=None,
        number_of_wan_stage_boundaries=1,
        persistent_connections=True,
        network_cost_confidence="unmeasured",
        network_cost_provenance="private protocol volume not yet observed",
        required_memory_bytes=100,
        score=1,
        explanation=("both physical hosts own a required tensor share",),
        engine_parameters={"tensor_split": {"node-a/worker": 0.6, "node-b/worker": 0.4}},
    )


def test_explain_plan_is_complete_and_preserves_unknown_network_cost() -> None:
    plan = _plan()
    reports = (
        EngineSupportReport(
            engine_id="llamacpp-rpc",
            status=EngineSupportStatus.SUPPORTED,
            reason="pinned runtime supports qwen3moe",
            model_architecture="qwen3_moe",
            model_format="gguf",
            required_runtime="pinned llama.cpp",
            required_features=("qwen3moe",),
        ),
        EngineSupportReport(
            engine_id="native-stage",
            status=EngineSupportStatus.UNSUPPORTED_FORMAT,
            reason="native Qwen3 MoE requires safetensors",
            model_architecture="qwen3_moe",
            model_format="gguf",
        ),
    )
    decision = CanonicalPlanningDecision(
        selected=plan,
        candidates=(plan,),
        engine_support=reports,
    )
    summary = ClusterRunSummary(
        run_id="run-fixture",
        status="dry-run",
        model_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        model_fingerprint=plan.model_fingerprint,
        model_architecture="qwen3_moe",
        model_architecture_source="config.architectures",
        model_format="gguf",
        total_model_size_bytes=100,
        variant="UD-Q4_K_M",
        quantization="UD-Q4_K_M",
        engine_id=plan.engine_id,
        execution_identity=plan.execution_identity,
        distributed_execution_required=True,
        distributed_execution_achieved=True,
        engine_support=reports,
        canonical_decision=decision,
        mode="capacity",
        plan=plan,
        started_at_unix_ns=1,
        completed_at_unix_ns=2,
        elapsed_seconds=0,
    )

    explanation = _plan_explanation(summary)

    assert explanation["model"] == {
        "id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "architecture": "qwen3_moe",
        "architecture_source": "config.architectures",
        "format": "gguf",
        "revision": "a" * 40,
        "quantization": "UD-Q4_K_M",
        "variant": "UD-Q4_K_M",
        "total_model_size_bytes": 100,
        "dense_or_moe": "unknown",
        "total_parameters": None,
        "active_parameters": None,
        "layers": None,
        "hidden_size": None,
        "experts": None,
        "active_experts": None,
        "shared_experts": None,
        "attention_architecture": "unknown",
        "attention_metadata": {},
        "tensor_layout": "unknown",
        "multimodal": False,
        "capabilities": [],
    }
    network = explanation["network_topology"]
    assert network["estimated_bytes_per_token"] is None
    assert network["estimated_network_operations_per_token"] is None
    assert network["network_cost_confidence"] == "unmeasured"
    assert explanation["distributed_execution_required"] is True
    assert explanation["distributed_execution_achieved_by_plan"] is True
    assert [item["engine"] for item in explanation["compatible_engines"]] == ["llamacpp-rpc"]
    assert [item["engine"] for item in explanation["rejected_engines"]] == ["native-stage"]
    assert (
        sum(item["model_bytes_owned"] for item in explanation["worker_memory_and_devices"]) == 100
    )
