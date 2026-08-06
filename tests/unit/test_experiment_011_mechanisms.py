from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from swarm_inference.experiments.experiment_010.transport import NETWORK_PROFILES
from swarm_inference.experiments.experiment_011.analysis import (
    PROFILE_ORDER,
    SUMMARY_COLUMNS,
    build_network_summary,
    profile_parameters_hash,
    write_csv,
)
from swarm_inference.experiments.experiment_011.charts import generate_network_charts
from swarm_inference.experiments.experiment_011.drafting import (
    PromptLookupDraftProvider,
    verify_greedy_candidates,
)
from swarm_inference.experiments.experiment_011.runner import (
    _reconstruct_expert_rpc_dependencies,
)
from swarm_inference.model.partition import (
    LayerCost,
    StageAssignment,
    StagePlan,
    balanced_ranges,
    equal_ranges,
)
from swarm_inference.runtime.telemetry import reconstruct_critical_path
from swarm_inference.transport.compression import (
    AdaptiveCompressionController,
    byte_shuffle,
    byte_unshuffle,
    compress_lossless,
    decompress_lossless,
)
from swarm_inference.transport.stage_tensor import (
    pack_tensor,
    tensor_raw_bytes,
    unpack_tensor,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _cost(layer_id: int, execution_ns: int) -> LayerCost:
    return LayerCost(
        layer_id=layer_id,
        execution_ns=execution_ns,
        weight_bytes=100,
        kv_bytes_per_token=8,
        peak_temporary_bytes=16,
        activation_bytes=8,
        measured=True,
    )


def test_equal_partition_is_complete_nonoverlapping_and_contiguous() -> None:
    ranges = equal_ranges(16, 4)
    assert ranges == ((0, 4), (4, 8), (8, 12), (12, 16))
    layers = [layer for start, end in ranges for layer in range(start, end)]
    assert layers == list(range(16))


def test_balanced_partition_minimises_contiguous_peak() -> None:
    costs = tuple(
        _cost(index, value)
        for index, value in enumerate([9_000, 1_000, 1_000, 1_000, 1_000, 1_000])
    )
    ranges = balanced_ranges(costs, 2, memory_limit_bytes=10_000)
    assert ranges == ((0, 1), (1, 6))


def test_stage_plan_serialization_and_ownership_validation(tmp_path: Path) -> None:
    assignments = (
        StageAssignment(0, 0, 2, (0, 1), 100, 10, 10, 8, 16, 8, "cuda:0", True, False, False),
        StageAssignment(1, 2, 4, (2, 3), 100, 10, 10, 8, 16, 8, "cuda:0", False, True, True),
    )
    plan = StagePlan(
        model_path="model",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        stage_count=2,
        layer_count=4,
        partition_method="equal",
        planner_objective="test",
        memory_limit_bytes=1_000,
        assignments=assignments,
        metadata_hash="hash",
    )
    path = tmp_path / "stage_plan.json"
    plan.write(path)
    assert StagePlan.read(path) == plan
    broken = replace(plan, assignments=(assignments[0], assignments[0]))
    with pytest.raises(ValueError, match="missing, duplicate, overlapping"):
        broken.validate()


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_byte_shuffle_and_compression_round_trip(width: int) -> None:
    payload = bytes(range(64)) * 64
    assert byte_unshuffle(byte_shuffle(payload, width), width) == payload
    compressed = compress_lossless(payload, mode="byte_shuffle_fast_codec", element_width=width)
    assert decompress_lossless(compressed).payload == payload


def test_strided_singleton_integer_tensor_has_canonical_wire_bytes() -> None:
    source = torch.arange(11, dtype=torch.int64).reshape(1, 11)[:, -1]
    assert source.stride() == (11,)
    assert tensor_raw_bytes(source) == (10).to_bytes(8, byteorder="little", signed=True)
    packed = pack_tensor(source, requested_mode="none")
    restored, _ = unpack_tensor(packed.payload, packed.attributes())
    assert torch.equal(restored, source)


def test_expert_rpc_serial_waits_are_reconstructed_from_request_events(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "session" / "worker-telemetry.jsonl"
    telemetry.parent.mkdir(parents=True)
    events = [
        {
            "event": "native_expert_request_completed",
            "worker_id": "worker-0",
            "request_id": f"run-colibri-{sequence}-token-{position}-layer-0-worker-0",
            "execution_sequence": sequence,
            "wall_time_ns": sequence,
            "duration_ns": 10,
            "bytes_received": 4,
            "bytes_sent": 8,
            "model_fingerprint": "sha256:model",
        }
        for sequence, position in ((1, 0), (2, 11))
    ]
    telemetry.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    result = _reconstruct_expert_rpc_dependencies(
        output_directory=tmp_path,
        generated_tokens=2,
        prompt_tokens=11,
        expected_message_count=2,
    )
    assert result["serial_waits_by_generated_token"] == [1, 1]
    assert result["serial_waits_per_token"] == 1
    assert result["not_estimated_from_aggregate_message_total"] is True
    assert result["valid"] is True


def test_adaptive_compression_uses_measured_cost_not_profile_name() -> None:
    controller = AdaptiveCompressionController(minimum_saving_ns=0)
    slow = controller.decide(
        raw_payload_bytes=1_000_000,
        compressed_payload_bytes=100_000,
        encode_ns=100_000,
        decode_ns=100_000,
        bandwidth_bps=10_000_000,
        rtt_ms=1,
        queue_delay_ms=0,
    )
    fast = controller.decide(
        raw_payload_bytes=1_000_000,
        compressed_payload_bytes=100_000,
        encode_ns=100_000,
        decode_ns=100_000,
        bandwidth_bps=100_000_000_000,
        rtt_ms=1,
        queue_delay_ms=0,
    )
    assert slow.selected_mode == "byte_shuffle_fast_codec"
    assert fast.selected_mode == "none"


def test_prompt_lookup_and_exact_candidate_rejection() -> None:
    provider = PromptLookupDraftProvider(minimum_match=2)
    assert provider.propose([1, 2, 3, 1, 2], depth=3) == [3, 1, 2]
    result = verify_greedy_candidates([3, 9, 5], [3, 4, 5])
    assert result.accepted_tokens == (3,)
    assert result.rejected_tokens == (9, 5)
    assert result.first_rejection_index == 1


def test_critical_path_reconstructed_from_links_not_message_total() -> None:
    events = [
        {
            "event": "socket_send_end",
            "data_plane": "ring",
            "message_type": "DECODE",
            "payload_bytes": 8,
            "wire_bytes": 16,
            "duration_ns": 2,
        },
        {
            "event": "socket_receive_end",
            "critical_dependency": True,
            "event_id": "receive",
            "unblocks_event_id": "compute",
            "monotonic_ns": 10,
            "dependency_token_position": 0,
            "token_position": 0,
            "source_stage": 0,
            "destination_stage": 1,
            "message_type": "DECODE",
            "duration_ns": 7,
        },
        {
            "event": "cuda_compute_start",
            "event_id": "compute",
            "monotonic_ns": 20,
            "stage_id": 1,
        },
    ]
    result = reconstruct_critical_path(events, generated_tokens=1)
    assert result["network_messages_total"] == 1
    assert result["serial_waits_total"] == 1
    assert result["dependency_edges"][0]["wait_ns"] == 7


def test_network_profile_manifest_is_reused_exactly() -> None:
    manifest_path = (
        REPOSITORY_ROOT / "tests" / "fixtures" / "experiment_010_transport_profiles.json"
    )
    archived = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = {name: profile.model_dump(mode="json") for name, profile in NETWORK_PROFILES.items()}
    assert archived == current
    assert [profile_parameters_hash(current[name]) for name in PROFILE_ORDER]


def test_synthetic_rows_are_not_accepted_as_measured_evidence(tmp_path: Path) -> None:
    rows = []
    for order, profile in enumerate(PROFILE_ORDER, start=1):
        for strategy, tps in (
            ("experiment_011_same_run_expert_rpc", 1.0),
            ("stage_ring_exact_2_equal", 2.0),
        ):
            rows.append(
                {
                    "profile_name": profile,
                    "strategy": strategy,
                    "throughput_tps": tps,
                    "valid_for_claims": True,
                    "token_match": True,
                    "evidence_category": "MECHANISM_ONLY",
                    "profile_order": order,
                }
            )
    summary, _ = build_network_summary(rows, output_directory=tmp_path)
    assert all(float(row["stage_exact_median_tps"]) == 0.0 for row in summary)


def test_chart_generation_reads_evidence_and_meets_dimensions(tmp_path: Path) -> None:
    rows = []
    for order, profile in enumerate(PROFILE_ORDER, start=1):
        rows.append(
            {
                "profile_order": order,
                "profile_name": profile,
                "archived_010_tps": 2.5 / order,
                "same_run_baseline_median_tps": 2.4 / order,
                "stage_exact_median_tps": 3.0 / (order**0.3),
                "stage_exact_ci_low": 2.9 / (order**0.3),
                "stage_exact_ci_high": 3.1 / (order**0.3),
                "difference_median_tps": 1.0,
                "difference_ci_low": 0.5,
                "difference_ci_high": 1.5,
                "percentage_improvement": 100,
                "throughput_multiple": 2.0,
                "classification": "IMPROVED",
                "selected_method": "balanced",
                "selected_stage_count": 2,
                "compression_selected": False,
                "speculation_selected": False,
                "serial_wait_reduction": 0.96,
                "message_reduction": 0.96,
                "payload_reduction": 0.99,
            }
        )
    source = tmp_path / "network_profile_summary.csv"
    write_csv(source, rows, SUMMARY_COLUMNS)
    inspection = generate_network_charts(source, tmp_path / "charts")
    assert inspection["all_minimum_dimensions_met"]
    assert len(inspection["charts"]) == 4
    for row in inspection["charts"]:
        assert Path(row["png"]).is_file()
        assert Path(row["png"]).with_suffix(".svg").is_file()
        assert Path(row["png"]).with_suffix(".pdf").is_file()
    with source.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 8
