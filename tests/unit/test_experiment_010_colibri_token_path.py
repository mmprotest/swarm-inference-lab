from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from swarm_inference.experiments.experiment_010.colibri_native import (
    whole_expert_ownership_from_banks,
)
from swarm_inference.experiments.experiment_010.colibri_token_path import (
    _route_ownership_counts,
    compare_colibri_numeric_traces,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT = REPOSITORY_ROOT / "integrations" / "colibri" / "patches"
NUMERIC_HEADER = struct.Struct("<8sIiiiiQ")


def _patch(name: str) -> str:
    return (PATCH_ROOT / name).read_text(encoding="utf-8")


def _correction_patches() -> str:
    return "\n".join(
        _patch(name)
        for name in (
            "0005-olmoe-shared-expert-runtime.patch",
            "0006-olmoe-external-expert-dispatch.patch",
            "0007-olmoe-native-microshards.patch",
            "0008-olmoe-memory-residency-telemetry.patch",
        )
    )


def _write_numeric_trace(
    path: Path,
    records: list[tuple[int, int, int, np.ndarray]],
) -> None:
    with path.open("wb") as handle:
        for kind, token_position, layer_id, values in records:
            array = np.asarray(values, dtype="<f4")
            if array.ndim != 2:
                raise ValueError("numeric trace fixture values must be two-dimensional")
            payload = array.tobytes()
            handle.write(
                NUMERIC_HEADER.pack(
                    b"COLNUM1\x00",
                    kind,
                    token_position,
                    layer_id,
                    array.shape[0],
                    array.shape[1],
                    len(payload),
                )
            )
            handle.write(payload)


def _paired_trace(tmp_path: Path, *, change_hidden: bool = False):
    local = tmp_path / "local.trace"
    remote = tmp_path / "remote.trace"
    local_records = [
        (3, 0, 0, np.array([[0.75, 0.25]], dtype=np.float32)),
        (1, 0, 0, np.array([[1.0, -2.0, 3.0]], dtype=np.float32)),
        (2, 0, -1, np.array([[0.5, 3.0, 1.0]], dtype=np.float32)),
    ]
    remote_records = [
        (kind, token, layer, values.copy()) for kind, token, layer, values in local_records
    ]
    if change_hidden:
        remote_records[1][3][0, 1] += np.float32(0.25)
    _write_numeric_trace(local, local_records)
    _write_numeric_trace(remote, remote_records)
    return compare_colibri_numeric_traces(
        prompt_id="fixture",
        local_trace=local,
        distributed_trace=remote,
        expected_token_ids=[1],
    )


def test_colibri_shared_expert_runtime_matches_local() -> None:
    patch = _correction_patches()
    assert "c/olmoe_expert_runtime.c" in patch
    assert "olmoe_expert_runtime_accumulate_view_with_workspace" in patch
    assert "shared local expert execution failed" in patch
    assert "test_olmoe_expert_runtime" in patch


def test_colibri_expert_worker_uses_shared_runtime() -> None:
    patch = _correction_patches()
    assert '#include "olmoe_expert_runtime.h"' in patch
    assert "olmoe_expert_runtime_execute_whole(state->runtime" in patch
    assert "olmoe_expert_runtime_execute_exact_contributions" in patch


def test_colibri_rpc_hook_called_inside_moe() -> None:
    patch = _patch("0006-olmoe-external-expert-dispatch.patch")
    assert "static void moe(Model *m, Layer *l, int layer" in patch
    assert "olmoe_external_dispatch_execute(" in patch
    assert "external OLMoE contributions" in patch


def test_colibri_remote_result_changes_moe_output(tmp_path: Path) -> None:
    rows, summary = _paired_trace(tmp_path, change_hidden=True)
    boundary = next(row for row in rows if row["record_kind"] == "post_moe_hidden_state")
    assert boundary["exact_fp32_identity"] is False
    assert boundary["maximum_absolute_error"] == 0.25
    assert summary["all_records_exact_fp32"] is False
    assert summary["first_divergent_layer"] == 0


def test_colibri_remote_result_consumed_by_step() -> None:
    patch = _patch("0006-olmoe-external-expert-dispatch.patch")
    assert "moe(m, l, i, nrm, S, pos_base, tmp);" in patch
    assert "x[j] += tmp[j]" in patch
    assert "remote_result_count += remote_results" in patch


def test_colibri_remote_generation_token_identity(tmp_path: Path) -> None:
    rows, summary = _paired_trace(tmp_path)
    logits = next(row for row in rows if row["record_kind"] == "pre_sampling_logits")
    assert logits["local_argmax"] == logits["distributed_argmax"] == 1
    assert logits["sampled_logit_error"] == 0.0
    assert summary["all_records_exact_fp32"] is True


def test_colibri_remote_router_identity(tmp_path: Path) -> None:
    rows, summary = _paired_trace(tmp_path)
    route = next(row for row in rows if row["record_kind"] == "router_weights_exact_fp32")
    assert route["exact_fp32_identity"] is True
    assert summary["router_weights_exact"] is True


def test_colibri_hybrid_plan_has_local_and_remote_owners(tmp_path: Path) -> None:
    local_bank = tmp_path / "local-bank"
    local_bank.mkdir()
    (local_bank / "ownership.json").write_text(
        '{"owned_experts":[{"layer_id":0,"expert_id":2}],"owned_microshards":[]}\n',
        encoding="utf-8",
    )
    ownership = whole_expert_ownership_from_banks([local_bank])
    assert ownership == [{"layer_id": 0, "expert_id": 2}]

    route = tmp_path / "route.trace"
    route.write_text("0 0 0 2:0.75 5:0.25\n", encoding="utf-8")
    assert _route_ownership_counts(route, {(0, 2)}) == (1, 1)

    duplicate = tmp_path / "duplicate-bank"
    duplicate.mkdir()
    (duplicate / "ownership.json").write_text(
        (local_bank / "ownership.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate local expert ownership"):
        whole_expert_ownership_from_banks([local_bank, duplicate])


def test_colibri_remote_weight_not_loaded_locally() -> None:
    patch = _patch("0006-olmoe-external-expert-dispatch.patch")
    assert "static void expert_get(Model *m, int layer, int eid, Slot **out)" in patch
    assert "olmoe_external_dispatch_forbidden_local_load" in patch
    assert "forbidden local expert load: remote-owned" in patch
    assert "exit(1);" in patch


def test_colibri_remote_failure_is_fail_closed() -> None:
    patch = _correction_patches()
    assert '"fail"' in patch
    assert "expert worker request failed without recovery" in patch
    assert "external expert dispatch failed at token=%d layer=%d" in patch


def test_colibri_exact_response_preserves_rank_order() -> None:
    patch = _patch("0006-olmoe-external-expert-dispatch.patch")
    assert "selected_rank_by_row" in patch
    assert "for (int rank = 0; rank < dispatch->top_k; rank++)" in patch
    assert "per_expert_exact" in patch
    assert "accumulate_exact_output" in patch


def test_colibri_fast_response_quality_contract() -> None:
    patch = _patch("0006-olmoe-external-expert-dispatch.patch")
    assert "per_worker_fast" in patch
    assert "quality_bounded" in patch
    assert "exact determinism requires per_expert_exact response mode" in patch


def test_native_microshard_generation_token_identity(tmp_path: Path) -> None:
    rows, summary = _paired_trace(tmp_path)
    assert len(rows) == 3
    assert summary["all_records_exact_fp32"] is True
    patch = _patch("0007-olmoe-native-microshards.patch")
    assert "execute_microshard_chain" in patch
    assert "fixed-order" in patch or "fixed_order" in patch


def test_capacity_end_to_end_generation() -> None:
    patch = _patch("0008-olmoe-memory-residency-telemetry.patch")
    assert "coordinator_owned_routed_expert_count" in patch
    assert "capacity-isolated coordinator: local routed-expert runtime disabled" in patch
    assert "local_expert_runtime_enabled" in patch


def test_numeric_trace_is_inside_the_real_step() -> None:
    patch = _patch("0008-olmoe-memory-residency-telemetry.patch")
    assert "COLI_SWARM_NUMERIC_TRACE" in patch
    assert "swarm_numeric_trace_write(1, pos_base, i, S, D, x)" in patch
    assert "swarm_numeric_trace_write(2, pos_base + S - 1" in patch
    assert "swarm_numeric_trace_write(3, token_position, layer, S, K, weights)" in patch
