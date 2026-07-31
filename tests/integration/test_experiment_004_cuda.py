from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.model_download
@pytest.mark.slow
def test_qwen3_fast_real_model_cuda_graph_batch_is_exact(
    repository_root: Path,
) -> None:
    """Opt-in RTX proof; the Experiment 004 run records the mandatory evidence."""

    if os.environ.get("SWARM_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("set SWARM_RUN_REAL_MODEL_TESTS=1 for the real Qwen3 fast-path test")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from huggingface_hub import snapshot_download

    from swarm_inference.model.qwen3_engine import load_qwen3_fast_engine
    from swarm_inference.model.qwen3_runtime import Qwen3EngineOptions

    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    model_path = Path(
        snapshot_download(
            "Qwen/Qwen3-0.6B",
            revision=revision,
            local_files_only=True,
        )
    )
    options = Qwen3EngineOptions.from_values(
        profile="qwen3_fast",
        attention_backend="auto",
        cache_backend="static",
        compile_mode="manual_cuda_graph",
        max_sequence_length=16,
        max_batch_size=2,
    )
    loaded = load_qwen3_fast_engine(
        model_id="Qwen/Qwen3-0.6B",
        model_revision=revision,
        model_path=model_path,
        options=options,
    )
    result = loaded.engine.generate_batch(
        torch.tensor([[1], [1]], dtype=torch.long),
        request_ids=("gpu-a", "gpu-b"),
        output_tokens=8,
        scheduler_policy="balanced",
    )
    reference = json.loads(
        (
            repository_root
            / "artifacts"
            / "benchmarks"
            / "experiment-004"
            / "prechange"
            / "legacy-baseline.json"
        ).read_text(encoding="utf-8")
    )["measured"][0]["output_token_ids"][:8]

    assert result.output_token_ids == [reference, reference]
    assert result.metrics.profile == "qwen3_fast"
    assert result.metrics.batch_size == 2
    assert result.metrics.batch_forward_count >= 8
    assert result.metrics.cuda_graph_verified is True
    assert result.metrics.cuda_graph_replay_count == 7
    assert result.metrics.full_logits_transferred is False
    assert result.metrics.device_to_host_bytes < result.metrics.full_logit_equivalent_bytes
    assert result.metrics.cuda_synchronisations == 1
