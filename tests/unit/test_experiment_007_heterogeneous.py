from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pytest

from swarm_inference.backends.artifacts import (
    compare_tokenizers,
    gguf_mapping_from_sidecar,
    validate_mapping,
)
from swarm_inference.backends.interfaces import InterfaceOnlyAdapter
from swarm_inference.experiments.background import BackgroundAdmissionController
from swarm_inference.experiments.heterogeneous_node_utility import (
    HeterogeneousOptions,
    _best_active_expert,
    _contribution_frontier,
    _failure_summary,
    classify_overall_status,
)
from swarm_inference.experiments.heterogeneous_projection import (
    NETWORK_PROFILES,
    MeasuredRoleTrace,
    availability_economics_rows,
    replay_measured_role,
)
from swarm_inference.experiments.sglang_measurement import parse_sglang_scheduler_log
from swarm_inference.experiments.speculative_trace import replay_lossless_trace
from swarm_inference.planner import (
    CanaryMeasurement,
    HeterogeneousPlanner,
    NodeRole,
    NonDegradationPolicy,
    PlannerObjective,
    RoleCandidate,
    UtilityNormalisation,
    planner_regret,
)
from swarm_inference.worker.abi import (
    BackendAdapter,
    BackendInterfaceEvidence,
    ResultClassification,
    TokenPayload,
    WorkerBenchmarkProfile,
    WorkerCapabilities,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
    WorkerProtocolVersion,
    tensor_payload_from_array,
)
from swarm_inference.worker.universal import UniversalWorkerClient, UniversalWorkerServer


def capabilities() -> WorkerCapabilities:
    return WorkerCapabilities(
        architecture="x86_64",
        operating_system="test",
        cpu_model="generic",
        physical_cpu_cores=4,
        logical_cpu_cores=8,
        system_memory_bytes=8 * 1024**3,
        supported_weight_formats=["test"],
        supported_activation_dtypes=["bfloat16", "float32"],
        supported_cache_dtypes=["bfloat16"],
        supported_collectives=[],
        maximum_weight_bytes=4 * 1024**3,
        maximum_cache_bytes=1024**3,
        maximum_batch_size=4,
        maximum_context_length=1024,
        measured_network_upload_bps=1e9,
        measured_network_download_bps=1e9,
        coordinator_latency_ms=0.1,
    )


class DummyAdapter(BackendAdapter):
    backend_id = "dummy"
    supported_jobs = frozenset({WorkerJobType.TARGET_DECODE})

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.cancelled: set[str] = set()

    def capabilities(self) -> WorkerCapabilities:
        return capabilities()

    def benchmark_profile(self) -> WorkerBenchmarkProfile:
        return WorkerBenchmarkProfile(model_load_seconds=0.1, warmup_seconds=0.01)

    async def execute(self, job: WorkerJob) -> WorkerJobResult:
        if self.fail:
            raise RuntimeError("backend exploded")
        delay = float(job.metadata.get("delay", 0))
        if delay:
            await asyncio.sleep(delay)
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=(
                WorkerJobStatus.CANCELLED
                if job.request_id in self.cancelled
                else WorkerJobStatus.ACCEPTED
            ),
            output_payload=TokenPayload(token_ids=[7]),
            classification=ResultClassification.MEASURED_CUDA,
        )

    async def cancel(self, request_id: str) -> bool:
        self.cancelled.add(request_id)
        return True


def job(*, role: WorkerJobType = WorkerJobType.TARGET_DECODE, deadline_ms: int = 1000) -> WorkerJob:
    return WorkerJob(
        job_id="job",
        request_id="request",
        role=role,
        model_id="model",
        model_revision="revision",
        input_payload=TokenPayload(token_ids=[1]),
        deadline_ms=deadline_ms,
    )


def test_protocol_version_negotiation_and_major_rejection() -> None:
    local = WorkerProtocolVersion(major=1, minor=3, capabilities={"jobs", "cancel"})
    peer = WorkerProtocolVersion(major=1, minor=1, capabilities={"jobs", "heartbeat"})
    agreed = local.negotiate(peer)
    assert agreed == WorkerProtocolVersion(major=1, minor=1, capabilities={"jobs"})
    assert local.negotiate(WorkerProtocolVersion(major=2, minor=0)) is None


def test_canonical_tensor_payload_preserves_boundary_identity() -> None:
    payload = tensor_payload_from_array(
        np.arange(8, dtype=np.float32).reshape(1, 2, 4),
        tensor_id="tensor",
        request_id="request",
        stage_id=2,
        token_position=17,
        sequence_length=2,
        model_revision="immutable",
        partition_hash="partition",
        route_generation=4,
    )
    decoded = payload.to_tensor()
    assert decoded.model_revision == "immutable"
    assert decoded.partition_hash == "partition"
    assert decoded.route_generation == 4
    assert decoded.token_position == 17
    np.testing.assert_array_equal(decoded.array, np.arange(8, dtype=np.float32).reshape(1, 2, 4))
    assert b"pickle" not in __import__("base64").b64decode(payload.data_base64).lower()


@pytest.mark.asyncio
async def test_unsupported_deadline_and_interface_only_rejections() -> None:
    adapter = DummyAdapter()
    rejected = adapter.admission_result(job(role=WorkerJobType.MOE_EXPERT))
    assert rejected is not None and rejected.status == WorkerJobStatus.UNSUPPORTED
    expired = job().model_copy(update={"created_at_unix_ms": time.time_ns() // 1_000_000 - 5000})
    rejected = adapter.admission_result(expired)
    assert rejected is not None and rejected.status == WorkerJobStatus.DEADLINE_IMPOSSIBLE
    interface = InterfaceOnlyAdapter(BackendInterfaceEvidence(backend_id="mlx"), capabilities())
    result = await interface.execute(job())
    assert result.status == WorkerJobStatus.UNSUPPORTED
    assert "physical_execution_unproven" in result.detail


@pytest.mark.asyncio
async def test_universal_server_heartbeat_reconnect_hash_cancel_failure_and_shutdown(
    tmp_path: Path,
) -> None:
    from swarm_inference.worker.abi import WorkerIdentity

    identity = WorkerIdentity(
        worker_id="worker",
        node_id="node",
        public_key="test",
        backend_id="dummy",
        protocol_version=WorkerProtocolVersion(
            capabilities={"jobs", "heartbeat", "cancel", "shard-hash"}
        ),
    )
    server = UniversalWorkerServer(adapter=DummyAdapter(), identity=identity)
    host, port = await server.start()
    task = asyncio.create_task(server.serve_until_shutdown())
    client = UniversalWorkerClient(host, port)
    agreed = await client.negotiate(
        WorkerProtocolVersion(capabilities={"jobs", "heartbeat", "cancel"})
    )
    assert agreed.capabilities == {"jobs", "heartbeat", "cancel"}
    assert (await client.identity()).worker_id == "worker"
    assert (await client.capabilities()).architecture == "x86_64"
    assert (await client.heartbeat())["heartbeat_count"] == 1
    # A new client proves reconnect works rather than retaining a Python object channel.
    assert (await UniversalWorkerClient(host, port).heartbeat())["heartbeat_count"] == 2
    shard = tmp_path / "shard.bin"
    shard.write_bytes(b"canonical shard")
    from swarm_inference.protocol.checksums import sha256_file

    assert await client.validate_shard(shard, sha256_file(shard))
    assert not await client.validate_shard(shard, "0" * 64)
    assert await client.cancel("request")
    assert await client.shutdown()
    await asyncio.wait_for(task, timeout=2)

    failure_server = UniversalWorkerServer(adapter=DummyAdapter(fail=True), identity=identity)
    failure_host, failure_port = await failure_server.start()
    failure_task = asyncio.create_task(failure_server.serve_until_shutdown())
    failure_client = UniversalWorkerClient(failure_host, failure_port)
    result = await failure_client.submit(job())
    assert result.status == WorkerJobStatus.BACKEND_FAILURE
    assert "backend exploded" in result.detail
    await failure_client.shutdown()
    await asyncio.wait_for(failure_task, timeout=2)


def _tokenizer_tree(root: Path) -> None:
    root.mkdir()
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"a": 0, "b": 1}},
                "added_tokens": [{"content": "<eos>", "id": 2, "special": True}],
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(json.dumps({"eos_token_id": 2}), encoding="utf-8")


def test_backend_artifact_mapping_and_tokenizer_identity(tmp_path: Path) -> None:
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    _tokenizer_tree(target)
    _tokenizer_tree(draft)
    comparison = compare_tokenizers(target, draft)
    assert comparison["status"] == "PASS"
    assert comparison["token_id_comparison_allowed"]
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-real-test")
    from swarm_inference.protocol.checksums import sha256_file

    identity = comparison["draft"]
    sidecar = gguf.with_suffix(".gguf.conversion.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_model_id": "Qwen/test",
                "source_revision": "a" * 40,
                "gguf_path": str(gguf),
                "conversion_command": "convert then quantize",
                "conversion_version": "b" * 40,
                "quantisation": "Q8_0",
                "gguf_sha256": sha256_file(gguf),
                "tokenizer_hash": identity["tokenizer_hash"],
                "vocabulary_hash": identity["vocabulary_hash"],
                "special_tokens_hash": identity["special_tokens_hash"],
            }
        ),
        encoding="utf-8",
    )
    mapping = gguf_mapping_from_sidecar(sidecar, canonical_partition_hash="c" * 64)
    assert validate_mapping(mapping)["status"] == "PASS"
    gguf.write_bytes(b"tampered")
    with pytest.raises(Exception, match="hash"):
        gguf_mapping_from_sidecar(sidecar, canonical_partition_hash="c" * 64)


def candidate(role: NodeRole, gain: float, latency: float = 0.0) -> RoleCandidate:
    return RoleCandidate(
        node_id="node",
        role=role,
        expected_verified_token_gain=gain,
        predicted_p95_latency_delta_ms=latency,
        predicted_interactive_throughput_delta=gain,
        predicted_memory_bytes=1,
        predicted_transfer_bytes=0,
        predicted_failure_cost=0,
        verification_cost=0,
        admission_risk=0,
        classification=ResultClassification.MEASURED_X86_CPU,
    )


def planner() -> HeterogeneousPlanner:
    return HeterogeneousPlanner(
        policy=NonDegradationPolicy(),
        normalisation=UtilityNormalisation(
            verified_tps_scale=10,
            interactive_latency_ms_scale=100,
            transfer_bytes_scale=1,
            failure_cost_scale=1,
            verification_cost_scale=1,
            admission_risk_scale=1,
        ),
    )


def test_planner_positive_harmful_idle_update_reassignment_and_regret() -> None:
    roles = [candidate(NodeRole.BACKGROUND_INFERENCE, 2), candidate(NodeRole.IDLE, 0)]
    decision = planner().select(
        roles,
        objective=PlannerObjective.BALANCED,
        baseline_p95_latency_ms=100,
        baseline_interactive_throughput=10,
        maximum_memory_bytes=100,
    )
    assert decision.selected_role == NodeRole.BACKGROUND_INFERENCE
    harmful = [candidate(NodeRole.CRITICAL_PATH_STAGE, -5, 20), candidate(NodeRole.IDLE, 0)]
    instance = planner()
    decision = instance.select(
        harmful,
        objective=PlannerObjective.INTERACTIVE_LATENCY,
        baseline_p95_latency_ms=100,
        baseline_interactive_throughput=10,
        maximum_memory_bytes=100,
    )
    assert decision.selected_role == NodeRole.IDLE
    assert (
        next(item for item in decision.ranking if item.role == "critical_path_stage").eligible
        is False
    )
    violation = CanaryMeasurement(
        node_id="node",
        role=NodeRole.CRITICAL_PATH_STAGE,
        measured_verified_tps_gain=-5,
        measured_p95_latency_delta_ms=20,
        measured_interactive_throughput_delta=-5,
        baseline_p95_latency_ms=100,
        baseline_interactive_throughput=10,
        measured_utility=-1,
        passed=False,
        classification=ResultClassification.MEASURED_X86_CPU,
    )
    reassigned = instance.monitor_and_reassign(
        violation,
        harmful,
        objective=PlannerObjective.BALANCED,
        maximum_memory_bytes=100,
    )
    assert reassigned is not None and reassigned.selected_role == NodeRole.IDLE
    regret = planner_regret(
        {NodeRole.IDLE: 0, NodeRole.BACKGROUND_INFERENCE: 1}, NodeRole.BACKGROUND_INFERENCE
    )
    assert regret["planner_regret_fraction"] == 0
    assert "raspberry" not in inspect.getsource(HeterogeneousPlanner).lower()


@pytest.mark.parametrize("draft_length", [1, 2, 4, 8])
def test_lossless_speculative_trace_all_lengths(draft_length: int) -> None:
    target = [1, 2, 3, 4, 5, 6, 7, 8]
    draft = [1, 9, 3, 4, 0, 6, 7, 8]
    result, evidence = replay_lossless_trace(
        target=target,
        draft=draft,
        draft_length=draft_length,
        prompt_id="prompt",
        category="code",
    )
    assert result["exact_output_identity"]
    assert result["speculative_token_ids"] == target
    assert evidence
    assert 0 <= result["acceptance_rate"] <= 1
    assert result["accepted_tokens_per_verification"] >= 0
    assert 1 <= result["target_work_per_committed_token"] <= draft_length + 1


def test_background_non_degradation_suspends_and_resumes() -> None:
    controller = BackgroundAdmissionController()
    assert not controller.observe_latency(100, 106)
    assert controller.suspended
    controller.resume()
    assert controller.observe_pressure(0.5)
    assert not controller.observe_pressure(0.95)
    controller.resume()
    assert not controller.observe_throughput(
        100,
        94,
        maximum_decrease_fraction=0.05,
    )


def test_sglang_scheduler_profile_parses_prefill_and_decode(tmp_path: Path) -> None:
    log = tmp_path / "sglang.log"
    log.write_text(
        "Prefill batch, #new-seq: 4, #new-token: 64, token usage: 0.02, "
        "#running-req: 1, #queue-req: 2, cuda graph: True\n"
        "Decode batch, #running-req: 16, #token: 100, token usage: 0.10, "
        "cuda graph: True, #queue-req: 0\n",
        encoding="utf-8",
    )
    profile = parse_sglang_scheduler_log(log, maximum_running_requests=64)
    assert profile["status"] == "PASS"
    assert profile["batch_size_maximum"] == 16
    assert profile["scheduler_occupancy_fraction_maximum"] == 0.25
    assert profile["kv_cache_token_usage_fraction_maximum"] == 0.10
    assert profile["queue_requests_maximum"] == 2


def test_best_expert_requires_real_cpu_dispatch() -> None:
    selected = _best_active_expert(
        [
            {
                "positive_contribution_pass": True,
                "selected_cpu_expert_calls": 0,
                "throughput_retained_fraction": 2.0,
            },
            {
                "positive_contribution_pass": True,
                "selected_cpu_expert_calls": 3,
                "throughput_retained_fraction": 1.5,
            },
        ]
    )
    assert selected is not None
    assert selected["selected_cpu_expert_calls"] == 3


def test_network_and_availability_preserve_negative_marginal_utility() -> None:
    trace = MeasuredRoleTrace(
        role=NodeRole.CRITICAL_PATH_STAGE,
        request_payload_bytes=1024,
        response_payload_bytes=1024,
        measured_compute_ms=200,
        verified_tokens=1,
        baseline_service_ms=100,
        measured_marginal_verified_tps_gain=-5,
    )
    replay = replay_measured_role(trace, NETWORK_PROFILES["localhost"])
    assert replay["projected_marginal_verified_tps_gain"] < 0
    assert not replay["viable"]
    rows = availability_economics_rows(
        [trace],
        acquisition_seconds={},
        conversion_seconds={},
        load_seconds={NodeRole.CRITICAL_PATH_STAGE: 2},
        warmup_seconds={NodeRole.CRITICAL_PATH_STAGE: 1},
        lease_durations_seconds=[30],
        all_roles=[NodeRole.CRITICAL_PATH_STAGE, NodeRole.IDLE],
    )
    critical = next(row for row in rows if row["role"] == "critical_path_stage")
    idle = next(row for row in rows if row["role"] == "idle")
    assert critical["time_to_first_useful_work_seconds"] == 3
    assert not critical["positive_for_lease"]
    assert critical["minimum_useful_lease_duration_seconds"] is None
    assert idle["role_measurement_status"] == "idle"


def test_contribution_labels_keep_projections_separate_and_pi_unproven() -> None:
    rows, positive = _contribution_frontier(
        mixed_rows=[
            {
                "route": "cuda-cpu-cuda-cpu",
                "throughput_change_fraction": -0.8,
            }
        ],
        speculative_rows=[],
        expert_rows=[],
        background_rows=[],
        audit_rows=[],
    )
    assert not positive
    assert rows[0]["critical_path"] == "harmful"
    projected = rows[1:]
    assert all(row["classification"] == "projected_device_profile" for row in projected)
    pi = next(row for row in projected if row["device_profile"] == "raspberry_pi_5_class")
    assert pi["raspberry_pi_performance"] == "unproven"
    assert all(
        pi[role] == "projected" for role in ("critical_path", "background_inference", "idle")
    )


def test_failure_reporting_uses_required_status_schema(tmp_path: Path) -> None:
    summary = _failure_summary(
        run_directory=tmp_path,
        phase_errors={"mandatory_infrastructure": "no GPU"},
        target_revision="a" * 40,
    )
    required = {
        "experiment_integrity_status",
        "universal_worker_abi_status",
        "sglang_backend_status",
        "cpu_rank_backend_status",
        "llamacpp_backend_status",
        "canonical_artifact_mapping_status",
        "mixed_backend_correctness_status",
        "forced_critical_path_status",
        "cpu_speculative_status",
        "cpu_expert_status",
        "cpu_background_status",
        "integrity_audit_status",
        "arm64_compatibility_status",
        "planner_prediction_status",
        "planner_non_degradation_status",
        "planner_regret_status",
        "positive_cpu_contribution_status",
        "overall_status",
    }
    assert required <= summary.keys()
    assert summary["overall_status"] == "FAIL"
    assert summary["positive_cpu_contribution_status"] == "FAIL"
    assert HeterogeneousOptions().smoke is False


def test_reporting_status_requires_measured_positive_inference() -> None:
    assert (
        classify_overall_status(
            infrastructure_pass=True,
            planner_and_core_pass=True,
            positive_cpu_contribution=False,
        )
        == "PARTIAL_PASS"
    )
    assert (
        classify_overall_status(
            infrastructure_pass=True,
            planner_and_core_pass=True,
            positive_cpu_contribution=True,
        )
        == "PASS"
    )
    assert (
        classify_overall_status(
            infrastructure_pass=False,
            planner_and_core_pass=True,
            positive_cpu_contribution=True,
        )
        == "FAIL"
    )
