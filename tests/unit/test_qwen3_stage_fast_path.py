from __future__ import annotations

from swarm_inference.execution.qwen3_stage import _engine_options
from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.qwen3_runtime import Qwen3CacheBackend, Qwen3CompileMode
from swarm_inference.protocol.stage_worker import LoadStageRequest


def _load_request(*, device: str, fast_path_mode: str) -> LoadStageRequest:
    return LoadStageRequest(
        worker_id="worker-a",
        request_id="load-qwen",
        model_id="Qwen/Qwen3-0.6B",
        model_revision="immutable-revision",
        tokenizer_revision="immutable-tokenizer",
        topology_id="topology-a",
        assignment=StageAssignment(
            stage_id=0,
            layer_start=0,
            layer_end=1,
            layer_ids=(0,),
            weight_bytes=1,
            estimated_compute_ns=1,
            measured_compute_ns=None,
            kv_cache_bytes_per_token=1,
            peak_temporary_bytes=0,
            activation_bytes=1,
            device=device,
            owns_embeddings=True,
            owns_final_norm=True,
            owns_output_projection=True,
        ),
        adapter_id="qwen3_dense",
        fast_path_id="qwen3_cuda",
        fast_path_mode=fast_path_mode,
        device=device,
        dtype="bfloat16",
    )


def test_forced_manual_cuda_graph_reaches_the_stage_executor() -> None:
    options, selected, fallback = _engine_options(
        _load_request(device="cuda:0", fast_path_mode="manual_cuda_graph"),
        config={"max_position_embeddings": 2048},
    )

    assert selected == "manual_cuda_graph"
    assert fallback is None
    assert options.compile_mode == Qwen3CompileMode.MANUAL_CUDA_GRAPH
    assert options.cache_backend == Qwen3CacheBackend.STATIC


def test_non_cuda_stage_fails_closed_to_eager() -> None:
    options, selected, fallback = _engine_options(
        _load_request(device="cpu", fast_path_mode="manual_cuda_graph"),
        config={"max_position_embeddings": 2048},
    )

    assert selected == "eager"
    assert options.compile_mode == Qwen3CompileMode.EAGER
    assert fallback == "manual_cuda_graph requires CUDA on this stage"
