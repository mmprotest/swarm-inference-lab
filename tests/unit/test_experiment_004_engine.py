from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_inference.config.models import OperationKind, QueueConfig
from swarm_inference.experiments.engine_environments import (
    ExternalEngineEnvironment,
)
from swarm_inference.experiments.engine_performance import (
    _report_artifact,
    parse_external_engine_result,
    validate_benchmark_fairness,
)
from swarm_inference.model.qwen3 import Qwen3StageModule
from swarm_inference.model.qwen3_cache import StaticStageKVCache
from swarm_inference.model.qwen3_engine import (
    Qwen3GroupedStageChain,
    estimate_cuda_graph_bundle_bytes,
)
from swarm_inference.model.qwen3_runtime import (
    AttentionBackend,
    AttentionBackendEvidence,
    CompileDiagnostics,
    Qwen3CompileMode,
    Qwen3EngineOptions,
    Qwen3ExecutionProfile,
    select_cuda_graph_bucket,
    validate_attention_backend,
)
from swarm_inference.model.qwen3_sampling import (
    SamplingParameters,
    sample_final_logits,
)
from swarm_inference.model.stage_module import (
    BatchExecutionMetadata,
    StageExecutionMetadata,
)
from swarm_inference.protocol.messages import ActivationMetadata, ActivationRequest
from swarm_inference.protocol.tensor_codec import (
    ActivationTensor,
    decode_tensor,
    encode_tensor,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.transport.tensor_paths import (
    PreallocatedTensorTransport,
    TensorPath,
)
from swarm_inference.worker.continuous_scheduler import ContinuousBatchScheduler
from swarm_inference.worker.execution import ExecutionEngine, QueuedExecution
from swarm_inference.worker.metrics import WorkerMetrics

torch = pytest.importorskip("torch")


class _GroupedCudaStage:
    execution_profile = "qwen3_fast"

    def __init__(self, index: int, *, first: bool, last: bool) -> None:
        self.stage_id = index
        self.required_memory_bytes = index + 1
        self.stage = SimpleNamespace(
            layer_start=index,
            layer_end=index + 1,
        )
        self.engine_options = Qwen3EngineOptions.from_values(
            profile="qwen3_fast",
            attention_backend="sdpa",
            max_sequence_length=8,
        )
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.torch = torch
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.embed_tokens = object() if first else None
        self.lm_head = object() if last else None
        self.attention_backend = "sdpa"
        self.attention_evidence = AttentionBackendEvidence(
            requested="sdpa",
            selected="sdpa",
        )
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def prefill_batch_cuda(
        self,
        input_tensors: torch.Tensor,
        metadata: BatchExecutionMetadata,
    ) -> torch.Tensor:
        assert metadata.batch_size == 2
        self.inputs.append(input_tensors)
        output = input_tensors + 1
        self.outputs.append(output)
        return output

    decode_batch_cuda = prefill_batch_cuda

    def prefill_cuda(self, input_tensor: torch.Tensor, metadata: object) -> torch.Tensor:
        return input_tensor + 1

    decode_cuda = prefill_cuda

    def sample_cuda(self, logits: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return logits

    def cancel(self, _request_id: str) -> None:
        return None

    def cancel_batch(self, _request_ids: tuple[str, ...]) -> None:
        return None

    def cache_bytes(self) -> int:
        return 0

    def state_summary(self) -> dict[str, object]:
        return {
            "compile_diagnostics": {
                "prefill_compiled": False,
                "decode_compiled": False,
                "fallback_used": False,
            },
            "cache_count": 0,
            "cache_bytes": 0,
            "caches": [],
            "fast_batch_forward_count": len(self.outputs),
        }


def test_grouped_stage_chain_uses_direct_tensor_boundaries_in_one_context() -> None:
    first = _GroupedCudaStage(0, first=True, last=False)
    second = _GroupedCudaStage(1, first=False, last=True)
    chain = Qwen3GroupedStageChain([first, second])  # type: ignore[list-item]
    metadata = BatchExecutionMetadata(
        requests=(
            StageExecutionMetadata(
                request_id="a",
                token_position=0,
                sequence_length=1,
            ),
            StageExecutionMetadata(
                request_id="b",
                token_position=0,
                sequence_length=1,
            ),
        )
    )

    output = chain.prefill_batch_cuda(torch.zeros((2, 1, 3)), metadata)

    assert torch.equal(output, torch.full((2, 1, 3), 2.0))
    assert second.inputs[0] is first.outputs[0]
    assert chain.topology() == {
        "tensor_path": "in_process_gpu",
        "logical_stage_count": 2,
        "physical_worker_process_count": 1,
        "cuda_context_count": 1,
    }
    assert chain.state_summary()["fast_batch_forward_count"] == 1


def _static_cache() -> StaticStageKVCache:
    return StaticStageKVCache(
        torch_module=torch,
        request_ids=("request-a", "request-b"),
        model_revision="immutable-test-revision",
        stage_id=2,
        layer_start=4,
        layer_end=6,
        route_generation=1,
        cache_generation=0,
        max_sequence_length=8,
        key_value_head_count=1,
        head_dimension=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def test_profile_selection_is_explicit_and_safe() -> None:
    correctness = Qwen3EngineOptions.from_values(profile="qwen3_correctness")
    fast = Qwen3EngineOptions.from_values(profile="qwen3_fast")

    assert correctness.profile == Qwen3ExecutionProfile.CORRECTNESS
    assert correctness.attention_backend == AttentionBackend.EAGER
    assert correctness.final_worker_sampling is False
    assert correctness.boundary_diagnostics is True
    assert fast.profile == Qwen3ExecutionProfile.FAST
    assert fast.attention_backend == AttentionBackend.AUTO
    assert fast.final_worker_sampling is True
    assert fast.boundary_diagnostics is False


def test_attention_backend_selection_rejects_unavailable_backend() -> None:
    availability = {
        "eager": True,
        "sdpa": True,
        "flash_attention_2": False,
        "flashinfer": False,
    }
    validate_attention_backend(AttentionBackend.SDPA, availability=availability)
    with pytest.raises(RuntimeError, match=r"flash_attention_2.*unsupported"):
        validate_attention_backend(
            AttentionBackend.FLASH_ATTENTION_2,
            availability=availability,
        )


def test_static_cache_positions_snapshot_rollback_and_cleanup() -> None:
    cache = _static_cache()
    keys = torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4).to(torch.bfloat16)
    values = (keys + 100).to(torch.bfloat16)
    positions = torch.tensor([0, 1], dtype=torch.long)

    cache.prepare_append(token_position=0, query_length=2)
    read_keys, read_values = cache.update(
        keys,
        values,
        layer_idx=4,
        cache_kwargs={"cache_position": positions},
    )
    cache.commit_append()

    assert cache.request_slots == {"request-a": 0, "request-b": 1}
    assert torch.equal(read_keys, keys)
    assert torch.equal(read_values, values)
    assert cache.sequence_length == 2
    assert cache.used_bytes > 0
    assert cache.allocation_count == 4

    cache.snapshot("prefill")
    cache.prepare_append(token_position=2, query_length=1)
    next_keys = torch.full((2, 1, 1, 4), 7, dtype=torch.bfloat16)
    cache.update(
        next_keys,
        next_keys,
        layer_idx=4,
        cache_kwargs={"cache_position": torch.tensor([2])},
    )
    cache.commit_append()
    cache.rollback(2)
    cache.restore("prefill")
    forked = cache.fork(request_ids=("fork-a", "fork-b"))
    assert torch.equal(forked.read(4)[0], keys)

    reserved = cache.reserved_bytes
    assert cache.delete() == reserved
    assert cache.delete() == 0
    assert cache.used_bytes == 0
    with pytest.raises(Exception, match="deleted"):
        cache.read(4)


def test_static_cache_rejects_wrong_decode_position() -> None:
    cache = _static_cache()
    with pytest.raises(Exception, match="position mismatch"):
        cache.prepare_append(token_position=1, query_length=1)


def test_final_worker_greedy_sampling_is_compact_and_diagnostic_is_opt_in() -> None:
    logits = torch.tensor(
        [[[1.0, 7.0, 3.0], [9.0, 2.0, 4.0]], [[0.0, 1.0, 8.0], [2.0, 6.0, 5.0]]],
        dtype=torch.bfloat16,
    )
    compact = sample_final_logits(
        torch,
        logits,
        parameters=SamplingParameters(),
        request_ids=("a", "b"),
    )
    assert compact.token_ids.tolist() == [0, 1]
    assert compact.full_logits is None
    assert compact.coordinator_payload_bytes() == 2 * (8 + 2)

    diagnostic = sample_final_logits(
        torch,
        logits,
        parameters=SamplingParameters(return_full_logits=True, diagnostic_top_k=2),
        request_ids=("a", "b"),
    )
    assert diagnostic.full_logits is not None
    assert diagnostic.top_token_ids.shape == (2, 2)
    assert diagnostic.coordinator_payload_bytes() > compact.coordinator_payload_bytes()


def test_remote_transport_preserves_bfloat16_bits_and_reuses_buffers() -> None:
    transport = PreallocatedTensorTransport(
        torch_module=torch,
        device="cpu",
        path=TensorPath.REMOTE_COMPATIBLE,
        profile="qwen3_fast",
    )
    source = torch.tensor(
        [[1.0, -2.5, 3.25], [4.5, 0.0, -0.125]],
        dtype=torch.bfloat16,
    )
    frame = transport.encode_remote(source)
    restored = transport.decode_remote(frame)
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored.view(torch.uint16), source.view(torch.uint16))
    assert transport.metrics.serialised_bytes == len(frame)

    direct = PreallocatedTensorTransport(
        torch_module=torch,
        device="cpu",
        path=TensorPath.IN_PROCESS_GPU,
        profile="qwen3_fast",
    )
    assert direct.transfer(source) is source
    assert direct.metrics.host_to_device_bytes == 0
    assert direct.metrics.device_to_host_bytes == 0


def test_continuous_scheduler_is_fair_and_reports_batch_occupancy() -> None:
    scheduler = ContinuousBatchScheduler(
        policy="throughput",
        max_batch_size=2,
        prefill_chunk_size=4,
        kv_token_capacity=64,
        starvation_limit_iterations=2,
    )
    admitted = time.monotonic_ns()
    for request_id in ("a", "b", "c"):
        scheduler.admit(
            request_id,
            prompt_tokens=4,
            output_tokens=2,
            admitted_ns=admitted,
        )

    served: set[str] = set()
    for _ in range(12):
        iteration = scheduler.next_iteration()
        if iteration.prefill_request_ids:
            scheduler.mark_prefill(
                iteration.prefill_request_ids,
                useful_tokens=4 * len(iteration.prefill_request_ids),
            )
        if iteration.decode_request_ids:
            served.update(iteration.decode_request_ids)
            scheduler.mark_decode(iteration.decode_request_ids)
        if scheduler.active_request_count == 0 and scheduler.waiting_request_count == 0:
            break
    assert served == {"a", "b", "c"}
    state = scheduler.state()
    assert state["metrics"]["iterations"] >= 4
    assert state["metrics"]["batch_size_distribution"]


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(1, 1), (3, 4), (16, 16), (33, 64)],
)
def test_cuda_graph_bucket_selection(batch_size: int, expected: int) -> None:
    assert select_cuda_graph_bucket(batch_size, (1, 2, 4, 8, 16, 32, 64)) == expected


def test_cuda_graph_memory_projection_rejects_paging_sized_bundle() -> None:
    gib = 1024**3
    budget = int(32 * gib * 0.88)
    qwen_06b = estimate_cuda_graph_bundle_bytes(
        model_parameter_bytes=int(1.2 * gib),
        batch_size=64,
        maximum_sequence_length=513,
        output_tokens=512,
        vocabulary_size=151_936,
        hidden_size=1_024,
        key_value_heads=8,
        head_dimension=128,
        layer_count=28,
        dtype_bytes=2,
    )
    qwen_4b = estimate_cuda_graph_bundle_bytes(
        model_parameter_bytes=int(7.6 * gib),
        batch_size=64,
        maximum_sequence_length=513,
        output_tokens=512,
        vocabulary_size=151_936,
        hidden_size=2_560,
        key_value_heads=8,
        head_dimension=128,
        layer_count=36,
        dtype_bytes=2,
    )

    assert qwen_06b["projected_bytes"] < budget
    assert qwen_4b["projected_bytes"] > budget
    assert qwen_4b["retained_hidden_bytes"] > qwen_06b["retained_hidden_bytes"]


def test_dynamic_cache_memory_fallback_group_is_cleaned_up() -> None:
    stage = object.__new__(Qwen3StageModule)
    stage._dynamic_cache_fallback_groups = set()
    cancelled: list[str] = []
    stage.cancel = cancelled.append
    request_ids = ("request-a", "request-b")

    stage.use_dynamic_cache_memory_fallback(request_ids)
    assert request_ids in stage._dynamic_cache_fallback_groups

    stage.cancel_batch(request_ids)
    assert cancelled == list(request_ids)
    assert request_ids not in stage._dynamic_cache_fallback_groups


def test_compile_failure_is_reported_and_falls_back() -> None:
    class FailingTorch:
        @staticmethod
        def compile(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("compiler unavailable")

    stage = object.__new__(Qwen3StageModule)
    stage.engine_options = Qwen3EngineOptions.from_values(
        profile="qwen3_fast",
        compile_mode=Qwen3CompileMode.DEFAULT,
    )
    stage.torch = FailingTorch()
    stage._layer_stack = lambda *args: args[0]
    stage._compiled_prefill = None
    stage._compiled_decode = None
    stage._compile_diagnostics = CompileDiagnostics(requested_mode=Qwen3CompileMode.DEFAULT.value)

    stage._configure_compile()

    assert stage._compiled_prefill is None
    assert stage._compiled_decode is None
    assert stage._compile_diagnostics.fallback_used is True
    assert "compiler unavailable" in str(stage._compile_diagnostics.fallback_reason)


def _benchmark_job(engine: str) -> dict[str, object]:
    return {
        "engine": engine,
        "profile": engine,
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "immutable",
        "model_dtype": "bfloat16",
        "input_token_ids": [[1]],
        "output_tokens": 4,
        "batch_size": 1,
        "repeats": 5,
        "warmup_requests": 3,
        "reference_output_token_ids": [[2, 3, 4, 5]],
        "pretokenized_inputs": True,
        "greedy": True,
        "complete_output_length_required": True,
    }


def test_benchmark_fairness_validation_detects_mismatched_inputs() -> None:
    custom = _benchmark_job("custom_fast")
    huggingface = _benchmark_job("huggingface_eager")
    evidence = validate_benchmark_fairness([custom, huggingface])
    assert evidence["status"] == "PASS"
    assert evidence["workload_hash"]

    huggingface["output_tokens"] = 3
    with pytest.raises(ValueError, match="output_tokens"):
        validate_benchmark_fairness([custom, huggingface])


def test_external_engine_result_parser_requires_complete_repeats() -> None:
    job = _benchmark_job("huggingface_eager")
    measured = [
        {
            "output_token_ids": [[2, 3, 4, 5]],
            "metrics": {"aggregate_verified_output_tokens_per_second": 42.0},
        }
        for _ in range(5)
    ]
    payload = {
        "engine": "huggingface_eager",
        "profile": "huggingface_eager",
        "status": "PASS",
        "worker_status": "completed",
        "measured_repeats": measured,
        "exact_reference_identity": True,
        "statistics": {
            "median": 42.0,
            "minimum": 41.0,
            "maximum": 43.0,
            "standard_deviation": 0.5,
            "coefficient_of_variation": 0.01,
        },
    }
    assert parse_external_engine_result(payload, job=job) is payload
    payload["measured_repeats"] = measured[:-1]
    with pytest.raises(ValueError, match="repeat count"):
        parse_external_engine_result(payload, job=job)


def test_environment_record_keeps_external_python_outside_project_venv(
    repository_root: Path,
) -> None:
    root = repository_root / "artifacts" / "engine-environments" / "huggingface"
    record = ExternalEngineEnvironment(
        engine="huggingface",
        kind="virtualenv",
        root=str(root),
        requested_version="4.57.6",
        status="PASS",
        python_executable=str(root / "Scripts" / "python.exe"),
        lock_path=str(root / "requirements.lock.txt"),
    )
    external_python = Path(str(record.python_executable)).resolve()
    assert external_python.is_relative_to(Path(record.root).resolve())
    assert not external_python.is_relative_to((repository_root / ".venv").resolve())
    assert record.lock_path is not None


class _BatchModule:
    stage_id = 0
    required_memory_bytes = 1
    execution_profile = "qwen3_fast"

    def __init__(self) -> None:
        self.calls = 0
        self.request_ids: tuple[str, ...] = ()

    def execute_batch(
        self,
        activations: np.ndarray,
        *,
        metadata: BatchExecutionMetadata,
        operation: OperationKind,
    ) -> np.ndarray:
        assert operation == OperationKind.DECODE
        self.calls += 1
        self.request_ids = metadata.request_ids
        return activations + 10


@pytest.mark.asyncio
async def test_worker_microbatch_executes_one_real_forward_for_two_requests() -> None:
    module = _BatchModule()
    shards = SimpleNamespace(module=lambda _stage_id: module)
    engine = ExecutionEngine(
        worker_id="worker",
        identity=WorkerIdentity.generate(),
        shards=shards,
        queue_config=QueueConfig(
            capacity=4,
            max_microbatch_size=2,
            max_microbatch_wait_ms=10,
        ),
        metrics=WorkerMetrics(worker_id="worker"),
    )
    loop = asyncio.get_running_loop()
    items: list[QueuedExecution] = []
    for index, request_id in enumerate(("a", "b")):
        activation = ActivationTensor(
            tensor_id=f"tensor-{request_id}",
            request_id=request_id,
            stage_id=0,
            token_position=4,
            sequence_length=1,
            array=np.asarray([[[index]]], dtype=np.float32),
        )
        request = ActivationRequest(
            metadata=ActivationMetadata(
                request_id=request_id,
                tensor_id=activation.tensor_id,
                stage_id=0,
                operation=OperationKind.DECODE,
                token_position=4,
                sequence_length=1,
                cache_generation=0,
                route_generation=1,
                model_id="test",
                model_revision="immutable",
            ),
            tensor_payload=encode_tensor(activation),
        )
        items.append(
            QueuedExecution(
                request=request,
                enqueued_at=time.perf_counter(),
                future=loop.create_future(),
            )
        )

    results = engine._execute_batch(items, module=module)

    assert module.calls == 1
    assert module.request_ids == ("a", "b")
    assert [float(decode_tensor(result.tensor_payload).array.item()) for result in results] == [
        10.0,
        11.0,
    ]


def test_report_artifact_uses_canonical_bounded_datasets_and_native_charts() -> None:
    summary = {
        "conclusion": "Measured conclusion.",
        "overall_status": "FAIL",
        "correctness_status": "PASS",
        "custom_batch_one_output_tokens_per_second": 120.0,
        "speedup_over_remeasured_baseline": 4.2,
        "fraction_of_fastest_successful_production_engine": 0.75,
        "maximum_custom_result_cv": 0.03,
    }
    engine_rows = [
        {
            "model_id": "Qwen/Qwen3-0.6B",
            "workload": "decode-focused",
            "concurrency": 1,
            "engine": "custom_fast",
            "profile": "qwen3_fast",
            "status": "PASS",
            "median_aggregate_output_tokens_per_second": 120.0,
            "median_decode_output_tokens_per_second": 140.0,
            "coefficient_of_variation": 0.03,
            "exact_reference_identity": True,
            "full_logit_equivalent_bytes": 1000,
            "coordinator_bound_bytes": 10,
        }
    ]
    artifact = _report_artifact(
        summary=summary,
        engine_rows=engine_rows,
        optimisation_ladder_rows=[
            {
                "optimisation": "gpu_native_dynamic",
                "profile": "qwen3_fast",
                "status": "PASS",
                "output_tokens_per_second": 80.0,
                "exact_reference_identity": True,
            }
        ],
        environments=[
            {
                "engine": "sglang",
                "status": "FAIL",
                "diagnostic": "Docker daemon unavailable",
            }
        ],
    )

    assert artifact["surface"] == "report"
    manifest = artifact["manifest"]
    assert manifest["blocks"][0]["body"] == f"# {manifest['title']}"
    assert any(block["type"] == "chart" for block in manifest["blocks"])
    assert all(
        chart["encodings"]["x"]["field"] and chart["encodings"]["y"]["field"]
        for chart in manifest["charts"]
    )
    assert all(isinstance(rows, list) for rows in artifact["snapshot"]["datasets"].values())
    assert artifact["snapshot"]["status"] == "ready"
