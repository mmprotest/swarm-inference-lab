"""Experiment 014 command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_014.census import build_checkpoint_census
from swarm_inference.experiments.experiment_014.compatibility import build_compatibility_matrix
from swarm_inference.experiments.experiment_014.conversation import (
    certify_conversation_semantics,
)
from swarm_inference.experiments.experiment_014.deployment import (
    build_execution_plan,
    run_logical_rehearsal,
)
from swarm_inference.experiments.experiment_014.distribution import (
    build_distribution_manifest,
    materialize_worker_package,
)
from swarm_inference.experiments.experiment_014.oracle import run_serial_oracle
from swarm_inference.experiments.experiment_014.placement import write_placement_artifacts
from swarm_inference.experiments.experiment_014.support import build_model_support_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    census = subparsers.add_parser("census", help="build the exact Kimi K3 tensor census")
    census.add_argument("--checkpoint", type=Path, required=True)
    census.add_argument("--output", type=Path, required=True)
    census.add_argument("--verify-payload-hashes", action="store_true")
    census.add_argument("--integrity-receipt", type=Path)
    census.add_argument("--hash-workers", type=int, default=1)
    support = subparsers.add_parser(
        "support-matrix", help="build the fail-closed 93-layer support matrix"
    )
    support.add_argument("--checkpoint", type=Path, required=True)
    support.add_argument("--engine-source", type=Path, required=True)
    support.add_argument("--output", type=Path, required=True)
    support.add_argument("--oracle-receipt", type=Path)
    support.add_argument("--placement-receipt", type=Path)
    oracle = subparsers.add_parser(
        "serial-oracle", help="run and validate the real-weight serial oracle"
    )
    oracle.add_argument("--checkpoint", type=Path, required=True)
    oracle.add_argument("--executable", type=Path, required=True)
    oracle.add_argument("--output-directory", type=Path, required=True)
    oracle.add_argument("--prompt", default="Hi")
    oracle.add_argument("--generated-tokens", type=int, default=2)
    oracle.add_argument("--layer-limit", type=int)
    oracle.add_argument("--timeout-seconds", type=float)
    placement = subparsers.add_parser(
        "placement", help="solve node count and emit exact tensor placement"
    )
    placement.add_argument("--checkpoint", type=Path, required=True)
    placement.add_argument("--manifest", type=Path, required=True)
    placement.add_argument("--solver-output", type=Path, required=True)
    placement.add_argument("--node-count", type=int, default=73)
    execution = subparsers.add_parser(
        "execution-plan", help="build the full 93-layer deployment DAG"
    )
    execution.add_argument("--placement", type=Path, required=True)
    execution.add_argument("--output", type=Path, required=True)
    rehearsal = subparsers.add_parser("rehearsal", help="run the exact logical cluster DAG")
    rehearsal.add_argument("--placement", type=Path, required=True)
    rehearsal.add_argument("--execution-plan", type=Path, required=True)
    rehearsal.add_argument("--output", type=Path, required=True)
    rehearsal.add_argument("--generations", type=int, default=2)
    compatibility = subparsers.add_parser(
        "sm86-audit", help="build the fail-closed RTX 3090 matrix"
    )
    compatibility.add_argument("--kimi-source", type=Path, required=True)
    compatibility.add_argument("--cuda-source", type=Path, required=True)
    compatibility.add_argument("--makefile", type=Path, required=True)
    compatibility.add_argument("--output", type=Path, required=True)
    compatibility.add_argument("--sm86-binary", type=Path)
    distribution = subparsers.add_parser(
        "distribution", help="build exact worker model distribution"
    )
    distribution.add_argument("--checkpoint", type=Path, required=True)
    distribution.add_argument("--placement", type=Path, required=True)
    distribution.add_argument("--output", type=Path, required=True)
    materialize = subparsers.add_parser("materialize-worker", help="extract one exact worker package")
    materialize.add_argument("--placement", type=Path, required=True)
    materialize.add_argument("--worker-id", required=True)
    materialize.add_argument("--source-directory", type=Path, required=True)
    materialize.add_argument("--distribution-manifest", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    conversation = subparsers.add_parser(
        "conversation", help="certify checkpoint-authoritative Kimi chat token IDs"
    )
    conversation.add_argument("--checkpoint", type=Path, required=True)
    conversation.add_argument("--tokenizer-json", type=Path, required=True)
    conversation.add_argument("--executable", type=Path, required=True)
    conversation.add_argument("--openai-server", type=Path, required=True)
    conversation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "census":
        receipt = build_checkpoint_census(
            args.checkpoint,
            args.output,
            verify_payload_hashes=args.verify_payload_hashes,
            integrity_receipt_path=args.integrity_receipt,
            hash_workers=args.hash_workers,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "support-matrix":
        receipt = build_model_support_matrix(
            args.checkpoint,
            args.engine_source,
            args.output,
            oracle_receipt_path=args.oracle_receipt,
            placement_receipt_path=args.placement_receipt,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "serial-oracle":
        receipt = run_serial_oracle(
            args.checkpoint,
            args.executable,
            args.output_directory,
            prompt=args.prompt,
            generated_tokens=args.generated_tokens,
            layer_limit=args.layer_limit,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "placement":
        receipt = write_placement_artifacts(
            args.checkpoint,
            args.manifest,
            args.solver_output,
            manifest_node_count=args.node_count,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "execution-plan":
        receipt = build_execution_plan(args.placement, args.output)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "rehearsal":
        receipt = run_logical_rehearsal(
            args.placement,
            args.execution_plan,
            args.output,
            generations=args.generations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "sm86-audit":
        receipt = build_compatibility_matrix(
            args.kimi_source,
            args.cuda_source,
            args.makefile,
            args.output,
            sm86_binary=args.sm86_binary,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 1
    if args.command == "distribution":
        receipt = build_distribution_manifest(args.checkpoint, args.placement, args.output)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "materialize-worker":
        distribution_manifest = json.loads(
            args.distribution_manifest.read_text(encoding="utf-8")
        )
        source_hashes = {
            name: row["sha256"]
            for name, row in distribution_manifest["source"]["shards"].items()
        }
        receipt = materialize_worker_package(
            args.placement,
            args.worker_id,
            args.source_directory,
            args.output,
            verify_source_hashes=source_hashes,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "conversation":
        receipt = certify_conversation_semantics(
            args.checkpoint,
            args.tokenizer_json,
            args.executable,
            args.openai_server,
            args.output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
