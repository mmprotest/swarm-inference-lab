from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from swarm_inference.experiments.experiment_010.transport import NETWORK_PROFILES
from swarm_inference.experiments.experiment_011 import MODEL_REVISION, TOKENIZER_REVISION
from swarm_inference.experiments.experiment_011.partition import (
    build_stage_plan,
    inspect_model_partition_metadata,
)
from swarm_inference.experiments.experiment_011.runtime import StageRingController

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f"
REFERENCE_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "runs"
    / "experiment-010-correction-work"
    / "phase-6"
    / "local-correctness-references"
    / "code-01"
    / "reference.json"
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("SWARM_RUN_EXPERIMENT_011_GPU_TESTS") != "1"
        or not torch.cuda.is_available(),
        reason="set SWARM_RUN_EXPERIMENT_011_GPU_TESTS=1 for real-model stage-ring tests",
    ),
]


@pytest.mark.parametrize("stage_count", [2, 4])
def test_exact_real_stage_decode_and_kv_ownership(tmp_path: Path, stage_count: int) -> None:
    workload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    prompt = [int(value) for value in workload["prompt_ids"]]
    expected = [int(value) for value in workload["full_ids"][len(prompt) :]][:2]
    metadata = inspect_model_partition_metadata(
        MODEL_PATH,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    plan = build_stage_plan(
        MODEL_PATH,
        metadata=metadata,
        stage_count=stage_count,
        method="equal",
        memory_limit_bytes=20_000_000_000,
    )
    controller = StageRingController(
        run_id=f"gpu-test-{stage_count}",
        plan=plan,
        network_profile=NETWORK_PROFILES["loopback_unshaped"],
        output_directory=tmp_path / f"stage-{stage_count}",
        timeout_s=120,
    )
    result = controller.run(prompt_token_ids=prompt, generated_token_count=2)
    assert list(result.generated_token_ids) == expected
    assert result.valid_for_claims
    assert result.critical_path["messages_per_token"] == stage_count
    assert result.critical_path["serial_waits_per_token"] <= stage_count
    assert len(set(result.stage_process_ids)) == stage_count
    owned = [
        name
        for stage in result.ownership
        for name in stage["parameter_names"]
        if name.startswith("model.layers.")
    ]
    assert len(owned) == len(set(owned))


def test_exact_compressed_real_stage_decode(tmp_path: Path) -> None:
    workload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    prompt = [int(value) for value in workload["prompt_ids"]]
    expected = [int(value) for value in workload["full_ids"][len(prompt) :]][:2]
    metadata = inspect_model_partition_metadata(
        MODEL_PATH,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    plan = build_stage_plan(
        MODEL_PATH,
        metadata=metadata,
        stage_count=2,
        method="equal",
        memory_limit_bytes=20_000_000_000,
    )
    result = StageRingController(
        run_id="gpu-test-compression",
        plan=plan,
        network_profile=NETWORK_PROFILES["global_wan"],
        output_directory=tmp_path / "compression",
        compression_request="byte_shuffle_fast_codec",
        timeout_s=120,
    ).run(prompt_token_ids=prompt, generated_token_count=2)
    assert list(result.generated_token_ids) == expected
    assert result.valid_for_claims
    assert "byte_shuffle_fast_codec" in result.compression_modes_used
