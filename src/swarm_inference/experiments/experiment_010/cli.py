"""Command-line entry point for Experiment 010.

This deliberately has no implicit model download or backend substitution.  The
PowerShell reproduction wrapper performs any explicitly requested Colibri build
before invoking this module.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from swarm_inference.experiments.experiment_010.runner import (
    Experiment010Options,
    run_experiment_010,
)
from swarm_inference.experiments.experiment_010.schemas import Experiment010Mode


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment-010",
        description="Run the hardware-in-the-loop virtual swarm closure experiment.",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode",
        choices=[item.value for item in Experiment010Mode],
        default=Experiment010Mode.QUICK.value,
    )
    parser.add_argument("--model-path-level-a")
    parser.add_argument("--model-path-level-a-source")
    parser.add_argument("--model-path-level-b")
    parser.add_argument("--expert-bank-root")
    parser.add_argument("--kimi-fixture-path")
    parser.add_argument("--colibri-path")
    parser.add_argument("--output-directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--correction-pass", action="store_true")
    parser.add_argument("--rebuild-colibri", action="store_true")
    parser.add_argument("--rebuild-expert-workers", action="store_true")
    parser.add_argument("--rebuild-cuda", action="store_true")
    parser.add_argument("--apply-bridge-patches", action="store_true")
    parser.add_argument("--topology")
    parser.add_argument("--network-profile")
    parser.add_argument("--configuration")
    parser.add_argument(
        "--response-mode",
        choices=("per_expert_exact", "per_worker_fast"),
        default="per_expert_exact",
    )
    parser.add_argument("--failure-matrix", action="store_true")
    parser.add_argument("--corruption-matrix", action="store_true")
    parser.add_argument("--require-complete-full-run", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--telemetry-level",
        choices=("off", "summary", "detailed", "trace"),
        default="detailed",
    )
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--skip-level-b", action="store_true")
    parser.add_argument("--skip-kimi-fixture", action="store_true")
    parser.add_argument("--model-path-frontier")
    parser.add_argument("--level-b-only", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(arguments)
    options = Experiment010Options(
        mode=Experiment010Mode(namespace.mode),
        model_path_level_a=_optional_path(namespace.model_path_level_a),
        model_path_level_a_source=_optional_path(namespace.model_path_level_a_source),
        model_path_level_b=_optional_path(namespace.model_path_level_b),
        expert_bank_root=_optional_path(namespace.expert_bank_root),
        kimi_fixture_path=_optional_path(namespace.kimi_fixture_path),
        colibri_path=_optional_path(namespace.colibri_path),
        output_directory=_optional_path(namespace.output_directory),
        resume=namespace.resume,
        correction_pass=namespace.correction_pass,
        rebuild_colibri=namespace.rebuild_colibri,
        rebuild_expert_workers=namespace.rebuild_expert_workers,
        rebuild_cuda=namespace.rebuild_cuda,
        apply_bridge_patches=namespace.apply_bridge_patches,
        topology=namespace.topology,
        network_profile=namespace.network_profile,
        configuration=namespace.configuration,
        response_mode=namespace.response_mode,
        failure_matrix=namespace.failure_matrix,
        corruption_matrix=namespace.corruption_matrix,
        require_complete_full_run=namespace.require_complete_full_run,
        allow_incomplete=namespace.allow_incomplete,
        repeats=namespace.repeats,
        telemetry_level=namespace.telemetry_level,
        skip_model_download=namespace.skip_model_download,
        skip_level_b=namespace.skip_level_b,
        skip_kimi_fixture=namespace.skip_kimi_fixture,
        model_path_frontier=_optional_path(namespace.model_path_frontier),
        level_b_only=namespace.level_b_only,
    )
    outcome = run_experiment_010(namespace.repository_root, options)
    print(
        json.dumps(
            {
                "bundle": str(outcome.bundle_path),
                "error": outcome.error,
                "mode": options.mode.value,
                "verdict": outcome.verdict.value,
            },
            sort_keys=True,
        )
    )
    return 1 if outcome.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
