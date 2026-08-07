"""Adapter-local regression for the original Experiment 010 OLMoE execution seam.

Universal acceptance uses architecture-neutral fixtures. This retained tiny checkpoint proves
that the legacy adapter still reaches canonical whole-expert and microshard machinery; it does
not give OLMoE product or planner status.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast
from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.config.models import Backend, WorkerRole
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorClient,
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.execution.expert import (
    ExpertWeights,
    expert_content_hash,
    safetensors_expert_ownership_entry,
    slice_expert_weights,
)
from swarm_inference.model.olmoe import inspect_olmoe_partition_metadata
from swarm_inference.model.product import ProductModelReference
from swarm_inference.protocol.messages import StreamEventType
from swarm_inference.protocol.product import ModelDeployRequest, ModelPlanRequest
from swarm_inference.worker.service import run_worker

MODEL_ID = "test/tiny-product-olmoe"
MODEL_REVISION = "tiny-product-revision"
PROMPT_TOKEN_IDS = [1, 2, 3]
PROMPT = "one two three"
PINNED_EXPECTED_TOKEN_IDS = [13, 8]
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
EXPERT_BUDGET_BYTES = 1024 * 1024
LAYER_COUNT = 2
EXPERT_COUNT = 4


@dataclass(frozen=True, slots=True)
class _ExpertWorkerLaunch:
    worker_id: str
    role: WorkerRole
    manifest_path: Path
    control_endpoint: str
    expert_endpoint: str


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def _reserve_endpoints(count: int) -> list[str]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            sockets.append(listener)
        return [f"127.0.0.1:{listener.getsockname()[1]}" for listener in sockets]
    finally:
        for listener in sockets:
            listener.close()


def _weights(module: torch.nn.Module) -> ExpertWeights:
    up = module.up_proj.weight.detach().to(dtype=torch.float32).numpy().copy()
    gate = module.gate_proj.weight.detach().to(dtype=torch.float32).numpy().copy()
    down = module.down_proj.weight.detach().to(dtype=torch.float32).numpy().copy()
    return ExpertWeights(
        up=up,
        gate=gate,
        down=down,
        content_hash=expert_content_hash(up, gate, down),
    )


def _save_microshard(path: Path, weights: ExpertWeights) -> None:
    np.savez(
        path,
        up=weights.up,
        gate=weights.gate,
        down=weights.down,
        hidden_start=np.asarray(weights.hidden_offset, dtype=np.int64),
        logical_intermediate_dimension=np.asarray(weights.logical_width, dtype=np.int64),
        native_format=np.asarray(weights.native_format),
    )


def _tiny_snapshot(tmp_path: Path) -> tuple[OlmoeForCausalLM, Path, str]:
    torch.manual_seed(5010)
    model = OlmoeForCausalLM(
        OlmoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=LAYER_COUNT,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            num_experts_per_tok=2,
            num_experts=EXPERT_COUNT,
            pad_token_id=0,
            eos_token_id=31,
        )
    ).eval()
    snapshot = tmp_path / "tiny-product-olmoe"
    model.save_pretrained(snapshot, safe_serialization=True, max_shard_size="1KB")
    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_commit_hash"] = MODEL_REVISION
    _write_json(config_path, config)
    vocabulary = {
        "<pad>": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        **{f"token-{index}": index for index in range(4, 30)},
        "<unk>": 30,
        "</s>": 31,
    }
    tokenizer_backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(snapshot)
    tokenizer_payload = (snapshot / "tokenizer.json").read_bytes()
    tokenizer_revision = "sha256:" + hashlib.sha256(tokenizer_payload).hexdigest()
    assert tokenizer(PROMPT, add_special_tokens=True)["input_ids"] == PROMPT_TOKEN_IDS
    return model, snapshot, tokenizer_revision


def _whole_worker(
    tmp_path: Path,
    *,
    snapshot: Path,
    model_fingerprint: str,
    quantization_fingerprint: str,
    endpoints: list[str],
) -> list[_ExpertWorkerLaunch]:
    worker_id = "whole-expert-worker"
    manifest_path = tmp_path / "whole-expert-manifest.json"
    entries = [
        safetensors_expert_ownership_entry(snapshot, layer_id=layer_id, expert_id=expert_id)
        for layer_id in range(LAYER_COUNT)
        for expert_id in range(EXPERT_COUNT)
    ]
    _write_json(
        manifest_path,
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_fingerprint": model_fingerprint,
            "quantization_fingerprint": quantization_fingerprint,
            "loader_type": "safetensors",
            "model_path": str(snapshot),
            "owned_experts": entries,
            "measured_service_rates": {"whole_expert_calls_per_second": 10_000.0},
        },
    )
    return [
        _ExpertWorkerLaunch(
            worker_id=worker_id,
            role=WorkerRole.WHOLE_EXPERT,
            manifest_path=manifest_path,
            control_endpoint=endpoints[0],
            expert_endpoint=endpoints[1],
        )
    ]


def _microshard_workers(
    tmp_path: Path,
    *,
    model: OlmoeForCausalLM,
    model_fingerprint: str,
    quantization_fingerprint: str,
    endpoints: list[str],
) -> list[_ExpertWorkerLaunch]:
    workers: list[_ExpertWorkerLaunch] = []
    for shard_id in range(2):
        worker_id = f"microshard-worker-{shard_id}"
        manifest_path = tmp_path / f"microshard-manifest-{shard_id}.json"
        owned_entries: list[dict[str, object]] = []
        owned_microshards: list[dict[str, object]] = []
        for layer_id, layer in enumerate(model.model.layers):
            for expert_id, module in enumerate(layer.mlp.experts):
                source = _weights(module)
                split = source.logical_width // 2
                hidden_start, hidden_end = (
                    (0, split) if shard_id == 0 else (split, source.logical_width)
                )
                shard = slice_expert_weights(
                    source,
                    hidden_start=hidden_start,
                    hidden_end=hidden_end,
                )
                shard_path = (
                    tmp_path / f"layer-{layer_id}-expert-{expert_id}-microshard-{shard_id}.npz"
                )
                _save_microshard(shard_path, shard)
                owned_entries.append(
                    {
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "path": str(shard_path),
                        "content_hash": shard.content_hash,
                    }
                )
                owned_microshards.append(
                    {
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "hidden_start": hidden_start,
                        "hidden_end": hidden_end,
                        "logical_intermediate_dimension": source.logical_width,
                        "content_hash": shard.content_hash,
                    }
                )
        assert all(
            int(item["hidden_end"]) - int(item["hidden_start"])
            < int(item["logical_intermediate_dimension"])
            for item in owned_microshards
        )
        _write_json(
            manifest_path,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_fingerprint": model_fingerprint,
                "quantization_fingerprint": quantization_fingerprint,
                "loader_type": "npz",
                "owned_experts": owned_entries,
                "owned_microshards": owned_microshards,
                "measured_service_rates": {
                    "microshard_calls_per_second": 10_000.0,
                    "reduction_calls_per_second": 100_000.0,
                },
            },
        )
        workers.append(
            _ExpertWorkerLaunch(
                worker_id=worker_id,
                role=WorkerRole.EXPERT_MICROSHARD,
                manifest_path=manifest_path,
                control_endpoint=endpoints[shard_id * 2],
                expert_endpoint=endpoints[shard_id * 2 + 1],
            )
        )
    return workers


async def _wait_for_workers(
    core: CoordinatorCore,
    tasks: list[asyncio.Task[None]],
    expected_worker_ids: set[str],
) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        for task in tasks:
            if task.done() and (error := task.exception()) is not None:
                raise RuntimeError("product worker failed during startup") from error
        registered = {item.worker_id for item in core.registry.workers()}
        if expected_worker_ids <= registered:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"workers did not register: {expected_worker_ids}")


@pytest.mark.parametrize("policy", ["whole-remote", "microshard-remote"])
@pytest.mark.asyncio
async def test_normal_product_submit_consumes_remote_expert_results(
    tmp_path: Path,
    policy: Literal["whole-remote", "microshard-remote"],
) -> None:
    model, snapshot, tokenizer_revision = _tiny_snapshot(tmp_path)
    metadata = inspect_olmoe_partition_metadata(
        snapshot,
        model_revision=MODEL_REVISION,
        tokenizer_revision=tokenizer_revision,
    )
    local_tokens: list[int] = []
    expected_input = torch.tensor([PROMPT_TOKEN_IDS], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(2):
            expected_token = int(model(input_ids=expected_input).logits[0, -1].argmax().item())
            local_tokens.append(expected_token)
            expected_input = torch.cat(
                [expected_input, torch.tensor([[expected_token]], dtype=torch.long)], dim=1
            )
    assert local_tokens == PINNED_EXPECTED_TOKEN_IDS

    config = ProductCoordinatorConfig(
        require_trusted_workers=False,
        worker_heartbeat_timeout_s=30,
        control_timeout_s=20,
        request_timeout_s=20,
        event_queue_capacity=64,
        token_ingress_capacity=64,
    )
    core = CoordinatorCore(product_config=config, state_directory=tmp_path / "coordinator")
    assert core.coordinator_identity is not None
    coordinator = CoordinatorRpcServer(core)
    coordinator_port = await coordinator.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    coordinator_fingerprint = core.coordinator_identity.public_key_fingerprint

    endpoint_count = 6 if policy == "whole-remote" else 8
    endpoints = _reserve_endpoints(endpoint_count)
    stage_endpoints = endpoints[:4]
    expert_endpoints = endpoints[4:]
    if policy == "whole-remote":
        expert_workers = _whole_worker(
            tmp_path,
            snapshot=snapshot,
            model_fingerprint=metadata.model_fingerprint,
            quantization_fingerprint=metadata.quantization_fingerprint,
            endpoints=expert_endpoints,
        )
    else:
        expert_workers = _microshard_workers(
            tmp_path,
            model=model,
            model_fingerprint=metadata.model_fingerprint,
            quantization_fingerprint=metadata.quantization_fingerprint,
            endpoints=expert_endpoints,
        )

    stop_event = asyncio.Event()
    worker_tasks = [
        asyncio.create_task(
            run_worker(
                coordinator_endpoint=coordinator_endpoint,
                listen_endpoint=stage_endpoints[stage_id * 2],
                advertised_endpoint=stage_endpoints[stage_id * 2],
                backend=Backend.TORCH_CPU,
                memory_limit_bytes=MEMORY_LIMIT_BYTES,
                identity_path=tmp_path / f"stage-worker-{stage_id}-identity.json",
                worker_id=f"stage-worker-{stage_id}",
                stage_runtime_enabled=True,
                data_listen_endpoint=stage_endpoints[stage_id * 2 + 1],
                data_advertised_endpoint=stage_endpoints[stage_id * 2 + 1],
                device="cpu",
                dtype="float32",
                configured_model_path=snapshot,
                stop_event=stop_event,
                upload_bandwidth_bytes_s=1_000_000_000,
                download_bandwidth_bytes_s=1_000_000_000,
                network_rates_measured=True,
                trusted_coordinator_fingerprint=coordinator_fingerprint,
                worker_roles={WorkerRole.CONTIGUOUS_STAGE},
            ),
            name=f"product-stage-worker-{stage_id}",
        )
        for stage_id in range(LAYER_COUNT)
    ]
    for item in expert_workers:
        worker_tasks.append(
            asyncio.create_task(
                run_worker(
                    coordinator_endpoint=coordinator_endpoint,
                    listen_endpoint=item.control_endpoint,
                    advertised_endpoint=item.control_endpoint,
                    backend=Backend.TORCH_CPU,
                    memory_limit_bytes=MEMORY_LIMIT_BYTES,
                    identity_path=tmp_path / f"{item.worker_id}-identity.json",
                    worker_id=item.worker_id,
                    stop_event=stop_event,
                    upload_bandwidth_bytes_s=1_000_000_000,
                    download_bandwidth_bytes_s=1_000_000_000,
                    network_rates_measured=True,
                    trusted_coordinator_fingerprint=coordinator_fingerprint,
                    worker_roles={item.role},
                    expert_manifest_path=item.manifest_path,
                    expert_data_listen_endpoint=item.expert_endpoint,
                    expert_data_advertised_endpoint=item.expert_endpoint,
                    expert_residency_budget_bytes=EXPERT_BUDGET_BYTES,
                    expert_cache_budget_bytes=EXPERT_BUDGET_BYTES,
                    expert_queue_capacity=16,
                ),
                name=item.worker_id,
            )
        )

    client = CoordinatorClient(coordinator_endpoint, timeout_s=30)
    try:
        await _wait_for_workers(
            core,
            worker_tasks,
            {
                *(f"stage-worker-{stage_id}" for stage_id in range(LAYER_COUNT)),
                *(item.worker_id for item in expert_workers),
            },
        )
        reference = ProductModelReference(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            tokenizer_revision=tokenizer_revision,
            dtype="float32",
        )
        planned = await client.plan_model(
            ModelPlanRequest(
                reference=reference,
                stage_count=LAYER_COUNT,
                partition_method="equal",
                max_sequence_tokens=16,
                expert_policy=policy,
                require_remote_experts=True,
                allow_expert_local_fallback=False,
            )
        )
        plan = planned.plan
        assert plan.stage_count == LAYER_COUNT
        assert [item.assignment.layer_ids for item in plan.assignments] == [(0,), (1,)]
        assert len(plan.expert_plans) == LAYER_COUNT
        assert all(item.require_remote_experts for item in plan.expert_plans)
        placements = [item for stage in plan.expert_plans for item in stage.placements]
        assert len(placements) == LAYER_COUNT * EXPERT_COUNT
        assert all(item.strategy == policy for item in placements)
        assert all(item.forced_remote and not item.local_fallback_permitted for item in placements)
        assert all(item.rejected for item in placements)
        if policy == "microshard-remote":
            assert all(
                len(item.worker_ids) == 2
                and all(
                    int(shard["hidden_end"]) - int(shard["hidden_start"])
                    < int(shard["logical_intermediate_dimension"])
                    for shard in item.microshards
                )
                for item in placements
            )

        deployment = await client.deploy_model(ModelDeployRequest(plan=plan))
        assert deployment.deployment.ready
        submit = await asyncio.to_thread(
            CliRunner().invoke,
            app,
            [
                "submit",
                "--coordinator",
                coordinator_endpoint,
                "--prompt",
                PROMPT,
                "--max-new-tokens",
                "2",
                "--seed",
                "0",
                "--model-id",
                MODEL_ID,
                "--model-revision",
                MODEL_REVISION,
                "--stream",
                "--json",
            ],
        )
        assert submit.exit_code == 0, submit.output
        events = json.loads(submit.output)
        generated = [
            event
            for event in events
            if event["event_type"] == StreamEventType.TOKEN_GENERATED.value
        ]
        assert [event["token_id"] for event in generated] == PINNED_EXPECTED_TOKEN_IDS
        expected_trace_name = (
            "remote_whole_expert_result_consumed"
            if policy == "whole-remote"
            else "remote_microshard_result_consumed"
        )
        for token_position, token_event in enumerate(generated):
            assert token_event["expert_trace"]
            assert {int(item["layer_id"]) for item in token_event["expert_trace"]} == {
                0,
                1,
            }
            assert all(
                item["event"] == expected_trace_name
                and f":token-{token_position}:" in str(item["request_id"])
                and item["request_bytes"] > 0
                and item["response_bytes"] > 0
                and str(item["result_hash"]).startswith("sha256:")
                and item["fallback_reason"] is None
                for item in token_event["expert_trace"]
            )
            assert token_event["expert_metrics"]["remote_expert_calls"] == len(
                token_event["expert_trace"]
            )
            assert token_event["expert_metrics"]["bytes_transferred"] == sum(
                int(item["request_bytes"]) + int(item["response_bytes"])
                for item in token_event["expert_trace"]
            )
            assert not any(
                "fallback" in str(item.get("event", "")) for item in token_event["expert_trace"]
            )
        completed = [
            event
            for event in events
            if event["event_type"] == StreamEventType.REQUEST_COMPLETED.value
        ]
        assert len(completed) == 1
        assert completed[0]["final_token_ids"] == PINNED_EXPECTED_TOKEN_IDS

        status = await client.status()
        assert status.expert_worker_count == len(expert_workers)
        assert status.expert_bytes_transferred > 0
        assert status.expert_cache_misses > 0
        assert status.expert_fallbacks == 0
        assert "fixed_order_fp32" in status.expert_reduction_modes
        if policy == "whole-remote":
            assert status.owned_experts == LAYER_COUNT * EXPERT_COUNT
            assert status.owned_microshards == 0
            assert status.remote_expert_calls > 0
            assert status.remote_microshard_calls == 0
        else:
            assert status.owned_experts == 0
            assert status.owned_microshards == 2 * LAYER_COUNT * EXPERT_COUNT
            assert status.remote_expert_calls == 0
            assert status.remote_microshard_calls > 0
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(
                asyncio.gather(*worker_tasks, return_exceptions=True),
                timeout=15,
            )
        except TimeoutError:
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        await client.close()
        await coordinator.stop(grace_s=0)
