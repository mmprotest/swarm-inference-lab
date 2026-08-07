from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from swarm_inference.execution.expert import (
    ExpertWeights,
    execute_expert,
    expert_content_hash,
    slice_expert_weights,
)
from swarm_inference.execution.microshard import MicroshardRange
from swarm_inference.execution.moe import (
    HybridMoeBackend,
    MicroshardRemoteBackend,
    MicroshardTarget,
    WholeExpertRemoteBackend,
    WholeExpertTarget,
)
from swarm_inference.execution.olmoe_stage import ContiguousOlmoeStage
from swarm_inference.model.olmoe import inspect_olmoe_partition_metadata
from swarm_inference.model.partition import StageAssignment, build_stage_plan
from swarm_inference.protocol.expert import (
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ExpertResponseMode,
)


class _TorchExpertClient:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    def execute(
        self, request: ExpertExecutionRequest, activation: np.ndarray
    ) -> tuple[Any, np.ndarray, dict[str, int]]:
        assert request.execution_mode == ExpertExecutionMode.WHOLE_EXPERT
        with torch.inference_mode():
            result = self.module(torch.from_numpy(activation)).detach().numpy()
        return (
            SimpleNamespace(
                status="ok",
                integrity=SimpleNamespace(
                    model_fingerprint=request.model_fingerprint,
                    expert_hashes=dict(request.expert_hashes),
                ),
            ),
            np.ascontiguousarray(result),
            {"request_bytes": activation.nbytes, "response_bytes": result.nbytes},
        )


class _NumpyMicroshardClient:
    def __init__(self, weights: ExpertWeights) -> None:
        self.weights = weights

    def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        down_accumulators: np.ndarray | None = None,
    ) -> tuple[Any, np.ndarray, dict[str, int]]:
        assert request.execution_mode == ExpertExecutionMode.MICROSHARD
        result = execute_expert(
            activation,
            self.weights,
            hidden_start=request.hidden_start,
            hidden_end=request.hidden_end,
        )
        assert request.response_mode == ExpertResponseMode.PER_EXPERT_EXACT
        accumulator = (
            np.zeros((request.batch_rows, 1, request.latent_dimension), dtype=np.float32)
            if down_accumulators is None
            else np.ascontiguousarray(down_accumulators, dtype=np.float32).copy()
        )
        accumulator[:, 0, :] += result
        return (
            SimpleNamespace(
                status="ok",
                integrity=SimpleNamespace(
                    model_fingerprint=request.model_fingerprint,
                    expert_hashes=dict(request.expert_hashes),
                ),
            ),
            accumulator,
            {"request_bytes": activation.nbytes, "response_bytes": accumulator.nbytes},
        )


def _expert_weights(module: torch.nn.Module) -> ExpertWeights:
    up = module.up_proj.weight.detach().to(dtype=torch.float32).numpy().copy()
    gate = module.gate_proj.weight.detach().to(dtype=torch.float32).numpy().copy()
    down = module.down_proj.weight.detach().to(dtype=torch.float32).numpy().copy()
    return ExpertWeights(
        up=up,
        gate=gate,
        down=down,
        content_hash=expert_content_hash(up, gate, down),
    )


@pytest.fixture
def tiny_olmoe(
    tmp_path: Path,
) -> tuple[OlmoeForCausalLM, Path, StageAssignment]:
    torch.manual_seed(1010)
    config = OlmoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        num_experts_per_tok=2,
        num_experts=4,
        pad_token_id=0,
        eos_token_id=31,
    )
    model = OlmoeForCausalLM(config).eval()
    snapshot = tmp_path / "tiny-olmoe"
    model.save_pretrained(snapshot, safe_serialization=True, max_shard_size="1KB")
    metadata = inspect_olmoe_partition_metadata(
        snapshot,
        model_revision="tiny-revision",
        tokenizer_revision="tiny-tokenizer",
    )
    assignment = build_stage_plan(
        snapshot,
        metadata=metadata,
        stage_count=1,
        method="equal",
        memory_limit_bytes=1_000_000,
        device="cpu",
    ).assignments[0]
    return model, snapshot, assignment


def _run(stage: ContiguousOlmoeStage, *, session_id: str) -> Any:
    stage.open_session(session_id)
    return stage.execute_prefill(
        session_id=session_id,
        token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        cache_position_start=0,
        request_id="exact-request",
        token_position=0,
        deadline_ns=time.time_ns() + 10_000_000_000,
    )


def test_canonical_olmoe_stage_consumes_remote_whole_experts_exactly(
    tiny_olmoe: tuple[OlmoeForCausalLM, Path, StageAssignment],
) -> None:
    model, snapshot, assignment = tiny_olmoe
    local = ContiguousOlmoeStage(
        model_path=snapshot,
        assignment=assignment,
        stage_count=1,
        device="cpu",
        dtype=torch.float32,
    )
    experts = model.model.layers[0].mlp.experts
    remote_keys = {(0, expert_id) for expert_id in range(len(experts))}

    def backend_factory(_local_modules: dict[tuple[int, int], torch.nn.Module]) -> HybridMoeBackend:
        whole = WholeExpertRemoteBackend(
            targets={
                (0, expert_id): WholeExpertTarget(
                    worker_id=f"whole-{expert_id}",
                    client=_TorchExpertClient(expert),
                )
                for expert_id, expert in enumerate(experts)
            },
            model_id="tiny-olmoe",
            model_revision="tiny-revision",
            model_fingerprint="tiny-model-fingerprint",
            quantization_fingerprint="tiny-quantization-fingerprint",
            topology_id="tiny-ring",
            route_generation=1,
        )
        return HybridMoeBackend(
            whole_remote=whole,
            placement={key: "whole-remote" for key in remote_keys},
            require_remote=True,
        )

    remote = ContiguousOlmoeStage(
        model_path=snapshot,
        assignment=assignment,
        stage_count=1,
        device="cpu",
        dtype=torch.float32,
        moe_backend_factory=backend_factory,
        remote_experts=remote_keys,
    )
    try:
        local_result = _run(local, session_id="local")
        remote_result = _run(remote, session_id="remote-whole")
        torch.testing.assert_close(
            remote_result.hidden_states, local_result.hidden_states, rtol=0, atol=0
        )
        torch.testing.assert_close(remote_result.logits, local_result.logits, rtol=0, atol=0)
        assert torch.equal(remote_result.sampled_token_ids, local_result.sampled_token_ids)
        assert remote_result.expert_events
        assert all(
            event["event"] == "remote_whole_expert_result_consumed"
            and event["request_bytes"] > 0
            and event["response_bytes"] > 0
            and event["result_hash"]
            and event["fallback_reason"] is None
            for event in remote_result.expert_events
        )
    finally:
        local.close()
        remote.close()


def test_canonical_olmoe_stage_consumes_native_microshards_and_preserves_token(
    tiny_olmoe: tuple[OlmoeForCausalLM, Path, StageAssignment],
) -> None:
    model, snapshot, assignment = tiny_olmoe
    local = ContiguousOlmoeStage(
        model_path=snapshot,
        assignment=assignment,
        stage_count=1,
        device="cpu",
        dtype=torch.float32,
    )
    experts = model.model.layers[0].mlp.experts
    remote_keys = {(0, expert_id) for expert_id in range(len(experts))}
    targets: dict[tuple[int, int], list[MicroshardTarget]] = {}
    for expert_id, expert in enumerate(experts):
        weights = _expert_weights(expert)
        split = weights.intermediate_dimension // 2
        shards = [
            slice_expert_weights(weights, hidden_start=0, hidden_end=split),
            slice_expert_weights(
                weights,
                hidden_start=split,
                hidden_end=weights.intermediate_dimension,
            ),
        ]
        targets[(0, expert_id)] = [
            MicroshardTarget(
                ownership=MicroshardRange(
                    worker_id=f"micro-{expert_id}-{shard_id}",
                    layer_id=0,
                    expert_id=expert_id,
                    hidden_start=shard.hidden_offset,
                    hidden_end=shard.hidden_offset + shard.intermediate_dimension,
                    logical_intermediate_dimension=weights.intermediate_dimension,
                    content_hash=shard.content_hash,
                ),
                client=_NumpyMicroshardClient(shard),
            )
            for shard_id, shard in enumerate(shards)
        ]

    def backend_factory(_local_modules: dict[tuple[int, int], torch.nn.Module]) -> HybridMoeBackend:
        micro = MicroshardRemoteBackend(
            targets=targets,
            model_id="tiny-olmoe",
            model_revision="tiny-revision",
            model_fingerprint="tiny-model-fingerprint",
            quantization_fingerprint="tiny-quantization-fingerprint",
            topology_id="tiny-ring",
            route_generation=1,
        )
        return HybridMoeBackend(
            microshard_remote=micro,
            placement={key: "microshard-remote" for key in remote_keys},
            require_remote=True,
        )

    remote = ContiguousOlmoeStage(
        model_path=snapshot,
        assignment=assignment,
        stage_count=1,
        device="cpu",
        dtype=torch.float32,
        moe_backend_factory=backend_factory,
        remote_experts=remote_keys,
    )
    try:
        local_result = _run(local, session_id="local")
        remote_result = _run(remote, session_id="remote-micro")
        torch.testing.assert_close(
            remote_result.hidden_states,
            local_result.hidden_states,
            rtol=2e-5,
            atol=2e-7,
        )
        assert torch.equal(remote_result.sampled_token_ids, local_result.sampled_token_ids)
        assert remote_result.expert_events
        assert all(
            event["event"] == "remote_microshard_result_consumed"
            and len(event["worker_ids"]) == 2
            and event["request_bytes"] > 0
            and event["response_bytes"] > 0
            and event["result_hash"]
            and event["fallback_reason"] is None
            for event in remote_result.expert_events
        )
    finally:
        local.close()
        remote.close()
