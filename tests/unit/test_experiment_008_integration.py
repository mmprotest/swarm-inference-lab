from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.experiments.experiment_008.acquisition import _safe_extract
from swarm_inference.experiments.experiment_008.analysis import build_ablation_rows
from swarm_inference.experiments.experiment_008.backend import (
    GenerationResult,
    TokenEvent,
    event_token_ids,
    probe_llama_server,
)
from swarm_inference.experiments.experiment_008.bundle import REQUIRED_FILES, EvidenceBundle
from swarm_inference.experiments.experiment_008.reporting import build_report
from swarm_inference.experiments.experiment_008.workloads import (
    build_decode_workload,
    build_long_context_workload,
    scheduling_features,
)


def test_backend_event_parser_and_stream_metrics_use_token_timestamps() -> None:
    assert event_token_ids({"tokens": [3, 4]}) == [3, 4]
    assert event_token_ids({"completion_probabilities": [{"id": 7}]}) == [7]
    result = GenerationResult(
        prompt_token_ids=[1, 2],
        output_token_ids=[3, 4, 5],
        content="abc",
        token_events=[
            TokenEvent(3, 0, 2_000_000),
            TokenEvent(4, 1, 3_000_000),
            TokenEvent(5, 2, 5_000_000),
        ],
        admitted_monotonic_ns=500_000,
        started_monotonic_ns=1_000_000,
        completed_monotonic_ns=6_000_000,
        timings={},
        stop_reason="limit",
        success=True,
    )
    assert result.time_to_first_token_ms == 1.0
    assert result.inter_token_latencies_ms == [1.0, 2.0]
    assert result.decode_tokens_per_second == pytest.approx(2 / 0.003)


def test_capability_probe_never_infers_dynamic_expert_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"fake")

    def fake_probe(_path: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        output = (
            "--n-gpu-layers --override-tensor --n-cpu-moe --parallel --flash-attn"
            if arguments == ["--help"]
            else "version 1"
        )
        return subprocess.CompletedProcess([str(executable), *arguments], 0, output, "")

    monkeypatch.setattr("swarm_inference.experiments.experiment_008.backend._run_probe", fake_probe)
    probe = probe_llama_server(executable)
    assert probe.capabilities.tensor_buffer_override
    assert probe.capabilities.cpu_moe
    assert not probe.capabilities.expert_routing_trace
    assert not probe.capabilities.per_expert_dynamic_residency
    assert not probe.capabilities.expert_prefetch


def test_workloads_are_deterministic_and_scheduler_contract_hides_domain_metadata() -> None:
    def tokenizer(text: str) -> list[int]:
        return list(range(len(text.split())))

    first = build_decode_workload(
        tokenizer,
        prompt_count=20,
        input_minimum=32,
        input_maximum=48,
        output_tokens=16,
        seed=8008,
    )
    second = build_decode_workload(
        tokenizer,
        prompt_count=20,
        input_minimum=32,
        input_maximum=48,
        output_tokens=16,
        seed=8008,
    )
    assert [row.text_sha256 for row in first] == [row.text_sha256 for row in second]
    assert len({row.domain for row in first}) == 7
    assert set(scheduling_features(first[0], batch_size=1, concurrency=2)) == {
        "prompt_length",
        "requested_generation_length",
        "batch_size",
        "concurrency",
    }


def test_long_context_builder_hits_exact_token_target_without_quadratic_calls() -> None:
    calls = 0

    def tokenizer(text: str) -> list[int]:
        nonlocal calls
        calls += 1
        return list(range(len(text.split())))

    prompts = build_long_context_workload(
        tokenizer,
        target_tokens=512,
        prompt_count=2,
        output_tokens=8,
        seed=8008,
    )
    assert all(len(prompt.token_ids) == 512 for prompt in prompts)
    assert calls < 20


def test_bundle_atomic_checkpoint_and_required_audit(tmp_path: Path) -> None:
    bundle = EvidenceBundle(tmp_path / "experiment_008", resume=False)
    bundle.write_json("sample.json", {"b": 2, "a": 1})
    assert json.loads((bundle.root / "sample.json").read_text(encoding="utf-8")) == {
        "a": 1,
        "b": 2,
    }
    bundle.complete_configuration("A")
    resumed = EvidenceBundle(bundle.root, resume=True)
    assert resumed.is_configuration_complete("A")
    for filename in REQUIRED_FILES:
        resumed.write_text(filename, "placeholder\n")
    assert resumed.audit_required()["complete"]


def test_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "bad")
    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract(archive, tmp_path / "destination")


def test_ablation_missing_measurements_remain_null_not_zero() -> None:
    rows = build_ablation_rows(
        [], token_identity_by_configuration={item: None for item in "ABCDEFG"}
    )
    assert len(rows) == 7
    assert all(row["decode_tokens_per_second"] is None for row in rows)
    assert all(row["status"] == "NOT_RUN" for row in rows)


def test_ablation_ttft_change_is_fractional_reduction_not_speedup() -> None:
    observations = [
        {
            "configuration": configuration,
            "workload": "prefill_32k",
            "status": "COMPLETED",
            "metrics": {"time_to_first_token_ms": ttft},
        }
        for configuration, ttft in (("A", 100.0), ("G", 75.0))
    ]
    rows = build_ablation_rows(
        observations,
        token_identity_by_configuration={item: None for item in "ABCDEFG"},
    )
    adaptive = next(row for row in rows if row["configuration"] == "G")
    assert adaptive["ttft_change_vs_stock"] == pytest.approx(0.25)


def test_generic_config_loader_accepts_experiment_008(repository_root: Path) -> None:
    config = load_experiment_config(
        repository_root / "configs" / "experiments" / "experiment_008_adaptive_moe.yaml"
    )
    assert config.execution_mode == "single-host-adaptive-moe-saturation"  # type: ignore[comparison-overlap]


def test_report_distinguishes_raw_mixed_throughput_from_verified_zero(tmp_path: Path) -> None:
    bundle = EvidenceBundle(tmp_path / "experiment_008", resume=False)
    bundle.write_json(
        "correctness_results.json",
        {
            "comparisons": [
                {
                    "configuration": "G",
                    "workload": "mixed",
                    "exact_token_identity": False,
                }
            ]
        },
    )
    bundle.write_csv(
        "benchmark_results.csv",
        [
            {
                "configuration": "G",
                "status": "COMPLETED",
                "workload": "mixed",
                "combined_generated_tokens_per_second": 40.0,
                "verification_status": "FAILED_TOKEN_IDENTITY",
            }
        ],
    )
    bundle.write_csv(
        "ablation_results.csv",
        [{"configuration": "G", "mixed_verified_tokens_per_second": 0.0}],
    )

    report = build_report(bundle.root)

    assert "raw generated throughput: not measured / 40.000 tok/s" in report
    assert "mixed verified throughput: not measured / 0.000 tok/s" in report
    assert "measured zero verified throughput reflects 0/1 exact mixed outputs" in report
    assert "target-runtime CPU/GPU overlap: not measured" in report
    assert "not measured%" not in report
