"""Command-line entry points for Experiment 011."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from swarm_inference.experiments.experiment_010.transport import NETWORK_PROFILES
from swarm_inference.experiments.experiment_011 import MODEL_REVISION, TOKENIZER_REVISION
from swarm_inference.experiments.experiment_011.partition import (
    build_stage_plan,
    inspect_model_partition_metadata,
)
from swarm_inference.experiments.experiment_011.reference import (
    compare_capture_trees,
    run_local_reference,
)
from swarm_inference.experiments.experiment_011.runner import (
    ExperimentOptions,
    run_experiment,
)
from swarm_inference.experiments.experiment_011.runtime import StageRingController

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f"
DEFAULT_REFERENCE_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "runs"
    / "experiment-010-correction-work"
    / "phase-6"
    / "local-correctness-references"
    / "code-01"
    / "reference.json"
)


def _stage_smoke(arguments: argparse.Namespace) -> int:
    model_path = Path(arguments.model_path).resolve()
    reference = json.loads(Path(arguments.reference).read_text(encoding="utf-8"))
    prompt_ids = [int(value) for value in reference["prompt_ids"]]
    expected = [int(value) for value in reference["full_ids"][len(prompt_ids) :]]
    metadata = inspect_model_partition_metadata(
        model_path,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    plan = build_stage_plan(
        model_path,
        metadata=metadata,
        stage_count=arguments.stage_count,
        method=arguments.partition_method,
        memory_limit_bytes=arguments.memory_limit_bytes,
    )
    output = Path(arguments.output).resolve()
    plan.write(output / "stage_plan.json")
    controller = StageRingController(
        run_id=arguments.run_id,
        plan=plan,
        network_profile=NETWORK_PROFILES[arguments.profile],
        output_directory=output,
        compression_request=arguments.compression,
        capture_boundaries=arguments.capture_boundaries,
    )
    result = controller.run(
        prompt_token_ids=prompt_ids,
        generated_token_count=arguments.tokens,
        session_id=f"{arguments.run_id}-session",
        request_id=f"{arguments.run_id}-request",
    )
    expected_slice = expected[: arguments.tokens]
    report = {
        **result.to_dict(),
        "expected_token_ids": expected_slice,
        "token_match": list(result.generated_token_ids) == expected_slice,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result.valid_for_claims and report["token_match"] else 1


def _exactness_smoke(arguments: argparse.Namespace) -> int:
    model_path = Path(arguments.model_path).resolve()
    output = Path(arguments.output).resolve()
    metadata = inspect_model_partition_metadata(
        model_path,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    plan = build_stage_plan(
        model_path,
        metadata=metadata,
        stage_count=arguments.stage_count,
        method=arguments.partition_method,
        memory_limit_bytes=arguments.memory_limit_bytes,
    )
    plan.write(output / "stage_plan.json")
    reference = run_local_reference(
        model_path=model_path,
        workload_reference_path=Path(arguments.reference),
        plan=plan,
        generated_token_count=arguments.tokens,
        output_directory=output / "reference_captures",
    )
    controller = StageRingController(
        run_id=arguments.run_id,
        plan=plan,
        network_profile=NETWORK_PROFILES[arguments.profile],
        output_directory=output / "distributed",
        compression_request=arguments.compression,
        capture_boundaries=True,
    )
    session_id = f"{arguments.run_id}-session"
    distributed = controller.run(
        prompt_token_ids=list(reference.prompt_token_ids),
        generated_token_count=arguments.tokens,
        session_id=session_id,
        request_id=f"{arguments.run_id}-request",
    )
    comparison = compare_capture_trees(
        local_capture_directory=output / "reference_captures",
        distributed_capture_directory=output / "distributed" / "captures",
        session_id=session_id,
        prompt_id=reference.prompt_id,
        reproduction_command="python -m swarm_inference.experiments.experiment_011 exactness-smoke",
    )
    token_match = distributed.generated_token_ids == reference.generated_token_ids
    summary = {
        "token_match": token_match,
        "local_token_ids": list(reference.generated_token_ids),
        "distributed_token_ids": list(distributed.generated_token_ids),
        "capture_exact": comparison["exact"],
        "capture_comparison_count": comparison["comparison_count"],
        "capture_mismatch_count": comparison["mismatch_count"],
        "missing": comparison["missing"],
        "maximum_absolute_difference_fp32": comparison["maximum_absolute_difference_fp32"],
        "maximum_relative_l2_error_fp32": comparison["maximum_relative_l2_error_fp32"],
        "critical_path": distributed.critical_path,
        "valid_for_claims": distributed.valid_for_claims,
        "errors": list(distributed.errors),
    }
    (output / "exactness_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "exactness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if token_match and comparison["exact"] and distributed.valid_for_claims else 1


def _run(arguments: argparse.Namespace) -> int:
    if arguments.network_only and arguments.exactness_only:
        raise ValueError("--network-only and --exactness-only are mutually exclusive")
    run_root, zip_path, verdict = run_experiment(
        ExperimentOptions(
            mode=arguments.mode,
            run_id=arguments.run_id,
            model_path=arguments.model_path,
            draft_model_path=arguments.draft_model_path,
            stage_counts=tuple(arguments.stage_counts),
            profile_names=tuple(arguments.profile_names),
            skip_speculation=arguments.skip_speculation,
            network_only=arguments.network_only,
            exactness_only=arguments.exactness_only,
            resume=arguments.resume,
        )
    )
    print(f"Completed evidence directory: {run_root}")
    print(f"Completed ZIP bundle: {zip_path}")
    print(f"Scientific verdict: {verdict}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 011 exact stage-ring harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("stage-smoke", help="run a bounded real-model stage-ring smoke")
    smoke.add_argument("--run-id", default="experiment-011-stage-smoke")
    smoke.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    smoke.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH))
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--stage-count", type=int, choices=(2, 4, 8), default=2)
    smoke.add_argument("--partition-method", choices=("equal", "balanced"), default="equal")
    smoke.add_argument("--profile", choices=tuple(NETWORK_PROFILES), default="loopback_unshaped")
    smoke.add_argument(
        "--compression",
        choices=("none", "byte_shuffle_fast_codec", "adaptive"),
        default="none",
    )
    smoke.add_argument("--tokens", type=int, default=4)
    smoke.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    smoke.add_argument("--capture-boundaries", action="store_true")
    smoke.set_defaults(handler=_stage_smoke)
    exactness = subparsers.add_parser(
        "exactness-smoke", help="compare a real stage ring with monolithic tensor captures"
    )
    exactness.add_argument("--run-id", default="experiment-011-exactness-smoke")
    exactness.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    exactness.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH))
    exactness.add_argument("--output", required=True)
    exactness.add_argument("--stage-count", type=int, choices=(2, 4, 8), default=2)
    exactness.add_argument("--partition-method", choices=("equal", "balanced"), default="equal")
    exactness.add_argument(
        "--profile", choices=tuple(NETWORK_PROFILES), default="loopback_unshaped"
    )
    exactness.add_argument(
        "--compression",
        choices=("none", "byte_shuffle_fast_codec", "adaptive"),
        default="none",
    )
    exactness.add_argument("--tokens", type=int, default=4)
    exactness.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    exactness.set_defaults(handler=_exactness_smoke)
    run = subparsers.add_parser("run", help="run the Experiment 011 evidence workflow")
    run.add_argument("--mode", choices=("full", "quick"), default="full")
    run.add_argument("--run-id")
    run.add_argument("--model-path")
    run.add_argument("--draft-model-path")
    run.add_argument("--stage-counts", type=int, nargs="+", choices=(2, 4, 8), default=[2, 4, 8])
    run.add_argument("--profile-names", nargs="+", choices=tuple(NETWORK_PROFILES), default=list(NETWORK_PROFILES))
    run.add_argument("--skip-speculation", action="store_true")
    run.add_argument("--network-only", action="store_true")
    run.add_argument("--exactness-only", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))
