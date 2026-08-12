"""Experiment 014 command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_014.census import build_checkpoint_census
from swarm_inference.experiments.experiment_014.compatibility import (
    build_compatibility_matrix,
    build_cuda_operation_matrix,
    certify_sm86_complete_kimi,
    certify_sm86_routed_expert,
)
from swarm_inference.experiments.experiment_014.conversation import (
    certify_conversation_semantics,
)
from swarm_inference.experiments.experiment_014.cuda import (
    benchmark_real_attnres,
    benchmark_real_dense_projection,
    benchmark_real_embedding,
    benchmark_real_final_norm,
    benchmark_real_kda_core,
    benchmark_real_kda_stage,
    benchmark_real_lm_head,
    benchmark_real_mla_stage,
    benchmark_real_moe_reduction,
    benchmark_real_mxfp4_expert,
    benchmark_real_router,
    benchmark_real_shared_expert,
)
from swarm_inference.experiments.experiment_014.deployment import (
    build_execution_plan,
    run_logical_rehearsal,
)
from swarm_inference.experiments.experiment_014.distribution import (
    build_distribution_manifest,
    materialize_worker_package,
)
from swarm_inference.experiments.experiment_014.full_cuda import (
    benchmark_streamed_cuda_graph,
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
    oracle.add_argument("--k3-idot", type=int, choices=(0, 1))
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
    cuda_operations = subparsers.add_parser(
        "cuda-operation-matrix", help="audit Kimi operation reuse against Colibri CUDA"
    )
    cuda_operations.add_argument("--kimi-source", type=Path, required=True)
    cuda_operations.add_argument("--cuda-source", type=Path, required=True)
    cuda_operations.add_argument("--makefile", type=Path, required=True)
    cuda_operations.add_argument("--output", type=Path, required=True)
    cuda_expert = subparsers.add_parser(
        "cuda-expert", help="benchmark one real Kimi MXFP4 expert on CUDA"
    )
    cuda_expert.add_argument("--checkpoint", type=Path, required=True)
    cuda_expert.add_argument("--cuda-library", type=Path, required=True)
    cuda_expert.add_argument("--reference-library", type=Path, required=True)
    cuda_expert.add_argument("--output", type=Path, required=True)
    cuda_expert.add_argument("--device", type=int, default=0)
    cuda_expert.add_argument("--layer", type=int, default=1)
    cuda_expert.add_argument("--expert", type=int, default=0)
    cuda_expert.add_argument("--warmup", type=int, default=5)
    cuda_expert.add_argument("--iterations", type=int, default=30)
    cuda_expert.add_argument("--batch", type=int, default=1)
    cuda_expert.add_argument("--seed", type=int, default=14025)
    cuda_expert.add_argument("--cuda-architecture", default="sm_120")
    cuda_expert.add_argument("--up-first", action="store_true")
    cuda_expert.add_argument("--cycle-id", default="H014-025b")
    cuda_expert.add_argument("--fuse-gate-up", action="store_true")
    sm86_cert = subparsers.add_parser(
        "sm86-certification", help="promote inspected real Kimi sm_86 operation evidence"
    )
    sm86_cert.add_argument("--operation-matrix", type=Path, required=True)
    sm86_cert.add_argument("--benchmark", type=Path, required=True)
    sm86_cert.add_argument("--router-benchmark", type=Path)
    sm86_cert.add_argument("--dense-benchmark", type=Path, action="append", default=[])
    sm86_cert.add_argument("--final-norm-benchmark", type=Path)
    sm86_cert.add_argument("--binary", type=Path, required=True)
    sm86_cert.add_argument("--cuobjdump", type=Path, required=True)
    sm86_cert.add_argument("--output", type=Path, required=True)
    sm86_full = subparsers.add_parser(
        "sm86-full-certification",
        help="certify all 11 Kimi CUDA classes on one exact sm_86 binary",
    )
    sm86_full.add_argument("--operation-matrix", type=Path, required=True)
    sm86_full.add_argument("--benchmark", type=Path, action="append", required=True)
    sm86_full.add_argument("--binary", type=Path, required=True)
    sm86_full.add_argument("--cuobjdump", type=Path, required=True)
    sm86_full.add_argument("--qualification", type=Path, required=True)
    sm86_full.add_argument("--output", type=Path, required=True)
    sm86_full.add_argument("--cycle-id", default="H014-025ae")
    cuda_router = subparsers.add_parser(
        "cuda-router", help="benchmark real Kimi top-16 routing on resident CUDA"
    )
    cuda_router.add_argument("--checkpoint", type=Path, required=True)
    cuda_router.add_argument("--cuda-library", type=Path, required=True)
    cuda_router.add_argument("--activation", type=Path, required=True)
    cuda_router.add_argument("--output", type=Path, required=True)
    cuda_router.add_argument("--device", type=int, default=0)
    cuda_router.add_argument("--layer", type=int, default=1)
    cuda_router.add_argument("--warmup", type=int, default=50)
    cuda_router.add_argument("--iterations", type=int, default=500)
    cuda_router.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_dense = subparsers.add_parser(
        "cuda-dense", help="benchmark a real Kimi grouped-int4 projection on CUDA"
    )
    cuda_dense.add_argument("--checkpoint", type=Path, required=True)
    cuda_dense.add_argument("--cuda-library", type=Path, required=True)
    cuda_dense.add_argument("--activation", type=Path, required=True)
    cuda_dense.add_argument("--output", type=Path, required=True)
    cuda_dense.add_argument("--device", type=int, default=0)
    cuda_dense.add_argument("--layer", type=int, default=1)
    cuda_dense.add_argument("--tensor-suffix", default="self_attn.f_a_proj.weight")
    cuda_dense.add_argument("--warmup", type=int, default=50)
    cuda_dense.add_argument("--iterations", type=int, default=500)
    cuda_dense.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_dense.add_argument("--cycle-id", default="H014-025m")
    cuda_norm = subparsers.add_parser(
        "cuda-final-norm", help="benchmark real Kimi final RMSNorm on CUDA"
    )
    cuda_norm.add_argument("--checkpoint", type=Path, required=True)
    cuda_norm.add_argument("--cuda-library", type=Path, required=True)
    cuda_norm.add_argument("--activation", type=Path, required=True)
    cuda_norm.add_argument("--output", type=Path, required=True)
    cuda_norm.add_argument("--device", type=int, default=0)
    cuda_norm.add_argument("--warmup", type=int, default=50)
    cuda_norm.add_argument("--iterations", type=int, default=500)
    cuda_norm.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_norm.add_argument("--cycle-id", default="H014-025o")
    cuda_embedding = subparsers.add_parser(
        "cuda-embedding", help="benchmark the full resident real Kimi BF16 embedding"
    )
    cuda_embedding.add_argument("--checkpoint", type=Path, required=True)
    cuda_embedding.add_argument("--cuda-library", type=Path, required=True)
    cuda_embedding.add_argument("--output", type=Path, required=True)
    cuda_embedding.add_argument("--device", type=int, default=0)
    cuda_embedding.add_argument("--warmup", type=int, default=50)
    cuda_embedding.add_argument("--iterations", type=int, default=500)
    cuda_embedding.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_embedding.add_argument("--cycle-id", default="H014-025q")
    cuda_lm_head = subparsers.add_parser(
        "cuda-lm-head", help="benchmark the production-int8 real Kimi LM head"
    )
    cuda_lm_head.add_argument("--checkpoint", type=Path, required=True)
    cuda_lm_head.add_argument("--cuda-library", type=Path, required=True)
    cuda_lm_head.add_argument("--trace", type=Path, required=True)
    cuda_lm_head.add_argument("--oracle-logits", type=Path, required=True)
    cuda_lm_head.add_argument("--output", type=Path, required=True)
    cuda_lm_head.add_argument("--trace-row", type=int, default=187)
    cuda_lm_head.add_argument("--logits-row", type=int, default=1)
    cuda_lm_head.add_argument("--device", type=int, default=0)
    cuda_lm_head.add_argument("--warmup", type=int, default=20)
    cuda_lm_head.add_argument("--iterations", type=int, default=100)
    cuda_lm_head.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_shared_expert = subparsers.add_parser(
        "cuda-shared-expert",
        help="benchmark a complete real Kimi grouped-int4 shared expert",
    )
    cuda_shared_expert.add_argument("--checkpoint", type=Path, required=True)
    cuda_shared_expert.add_argument("--cuda-library", type=Path, required=True)
    cuda_shared_expert.add_argument("--activation", type=Path, required=True)
    cuda_shared_expert.add_argument("--output", type=Path, required=True)
    cuda_shared_expert.add_argument("--layer", type=int, default=1)
    cuda_shared_expert.add_argument("--device", type=int, default=0)
    cuda_shared_expert.add_argument("--warmup", type=int, default=20)
    cuda_shared_expert.add_argument("--iterations", type=int, default=100)
    cuda_shared_expert.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_moe_reduction = subparsers.add_parser(
        "cuda-moe-reduction",
        help="benchmark fixed-order reduction of 16 real Kimi expert outputs",
    )
    cuda_moe_reduction.add_argument("--checkpoint", type=Path, required=True)
    cuda_moe_reduction.add_argument("--cuda-library", type=Path, required=True)
    cuda_moe_reduction.add_argument("--reference-library", type=Path, required=True)
    cuda_moe_reduction.add_argument("--activation", type=Path, required=True)
    cuda_moe_reduction.add_argument("--output", type=Path, required=True)
    cuda_moe_reduction.add_argument("--layer", type=int, default=1)
    cuda_moe_reduction.add_argument("--device", type=int, default=0)
    cuda_moe_reduction.add_argument("--warmup", type=int, default=50)
    cuda_moe_reduction.add_argument("--iterations", type=int, default=500)
    cuda_moe_reduction.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_attnres = subparsers.add_parser(
        "cuda-attnres",
        help="benchmark real Kimi AttnRes against retained full-model decode state",
    )
    cuda_attnres.add_argument("--checkpoint", type=Path, required=True)
    cuda_attnres.add_argument("--cuda-library", type=Path, required=True)
    cuda_attnres.add_argument("--trace", type=Path, required=True)
    cuda_attnres.add_argument("--output", type=Path, required=True)
    cuda_attnres.add_argument("--trace-step", type=int, default=2)
    cuda_attnres.add_argument("--token-id", type=int, default=11)
    cuda_attnres.add_argument("--device", type=int, default=0)
    cuda_attnres.add_argument("--warmup", type=int, default=50)
    cuda_attnres.add_argument("--iterations", type=int, default=500)
    cuda_attnres.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_attnres.add_argument("--cycle-id", default="H014-025v")
    cuda_attnres.add_argument("--baseline", type=Path)
    cuda_kda_core = subparsers.add_parser(
        "cuda-kda-core",
        help="benchmark the stateful real-weight-derived Kimi KDA core",
    )
    cuda_kda_core.add_argument("--checkpoint", type=Path, required=True)
    cuda_kda_core.add_argument("--cuda-library", type=Path, required=True)
    cuda_kda_core.add_argument("--activation", type=Path, required=True)
    cuda_kda_core.add_argument("--trace", type=Path, required=True)
    cuda_kda_core.add_argument("--output", type=Path, required=True)
    cuda_kda_core.add_argument("--layer", type=int, default=1)
    cuda_kda_core.add_argument("--device", type=int, default=0)
    cuda_kda_core.add_argument("--warmup", type=int, default=20)
    cuda_kda_core.add_argument("--iterations", type=int, default=100)
    cuda_kda_core.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_kda_stage = subparsers.add_parser(
        "cuda-kda-stage",
        help="benchmark the complete stateful real Kimi KDA attention stage",
    )
    cuda_kda_stage.add_argument("--checkpoint", type=Path, required=True)
    cuda_kda_stage.add_argument("--cuda-library", type=Path, required=True)
    cuda_kda_stage.add_argument("--activation", type=Path, required=True)
    cuda_kda_stage.add_argument("--trace", type=Path, required=True)
    cuda_kda_stage.add_argument("--output", type=Path, required=True)
    cuda_kda_stage.add_argument("--layer", type=int, default=1)
    cuda_kda_stage.add_argument("--device", type=int, default=0)
    cuda_kda_stage.add_argument("--warmup", type=int, default=20)
    cuda_kda_stage.add_argument("--iterations", type=int, default=100)
    cuda_kda_stage.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_kda_stage.add_argument("--cycle-id", default="H014-025z")
    cuda_mla_stage = subparsers.add_parser(
        "cuda-mla-stage",
        help="benchmark the complete stateful real Kimi Gated MLA attention stage",
    )
    cuda_mla_stage.add_argument("--checkpoint", type=Path, required=True)
    cuda_mla_stage.add_argument("--cuda-library", type=Path, required=True)
    cuda_mla_stage.add_argument("--trace", type=Path, required=True)
    cuda_mla_stage.add_argument("--output", type=Path, required=True)
    cuda_mla_stage.add_argument("--layer", type=int, default=3)
    cuda_mla_stage.add_argument("--device", type=int, default=0)
    cuda_mla_stage.add_argument("--warmup", type=int, default=20)
    cuda_mla_stage.add_argument("--iterations", type=int, default=100)
    cuda_mla_stage.add_argument("--cuda-architecture", default="sm_86+c86_ptx")
    cuda_mla_stage.add_argument("--cycle-id", default="H014-025ac")
    cuda_streamed_graph = subparsers.add_parser(
        "cuda-streamed-graph",
        help="execute a real checkpoint-backed Kimi graph prefix on CUDA",
    )
    cuda_streamed_graph.add_argument("--checkpoint", type=Path, required=True)
    cuda_streamed_graph.add_argument("--cuda-library", type=Path, required=True)
    cuda_streamed_graph.add_argument("--oracle-trace", type=Path, required=True)
    cuda_streamed_graph.add_argument("--oracle-routes", type=Path, required=True)
    cuda_streamed_graph.add_argument("--oracle-logits", type=Path)
    cuda_streamed_graph.add_argument("--output", type=Path, required=True)
    cuda_streamed_graph.add_argument("--layer-limit", type=int, default=4)
    cuda_streamed_graph.add_argument("--prompt-token-id", type=int, action="append", default=None)
    cuda_streamed_graph.add_argument("--decode-token-id", type=int, default=11)
    cuda_streamed_graph.add_argument("--device", type=int, default=0)
    cuda_streamed_graph.add_argument("--relative-error-gate", type=float, default=3e-3)
    cuda_streamed_graph.add_argument("--cycle-id", default="H014-025af")
    cuda_streamed_graph.add_argument("--oracle-layer-count", type=int)
    persistent_final = subparsers.add_parser(
        "persistent-final-stage",
        help="benchmark one canonical resident real-Kimi final CUDA stage",
    )
    persistent_final.add_argument("--checkpoint", type=Path, required=True)
    persistent_final.add_argument("--cuda-library", type=Path, required=True)
    persistent_final.add_argument("--oracle-trace", type=Path, required=True)
    persistent_final.add_argument("--oracle-routes", type=Path, required=True)
    persistent_final.add_argument("--oracle-logits", type=Path, required=True)
    persistent_final.add_argument("--output", type=Path, required=True)
    persistent_final.add_argument("--registered", action="store_true")
    persistent_final.add_argument("--identity-manifest", type=Path)
    persistent_final.add_argument("--cycle-id")
    persistent_nonfinal = subparsers.add_parser(
        "persistent-nonfinal-stages",
        help="benchmark registered persistent KDA and Gated-MLA Kimi stages",
    )
    persistent_nonfinal.add_argument("--checkpoint", type=Path, required=True)
    persistent_nonfinal.add_argument("--cuda-library", type=Path, required=True)
    persistent_nonfinal.add_argument("--oracle-trace", type=Path, required=True)
    persistent_nonfinal.add_argument("--oracle-routes", type=Path, required=True)
    persistent_nonfinal.add_argument("--identity-manifest", type=Path, required=True)
    persistent_nonfinal.add_argument("--final-regression", type=Path, required=True)
    persistent_nonfinal.add_argument("--output", type=Path, required=True)
    persistent_nonfinal.add_argument("--cycle-id", default="H014-026c")
    persistent_stage_zero = subparsers.add_parser(
        "persistent-stage-zero",
        help="benchmark the registered persistent Kimi embedding+dense first stage",
    )
    persistent_stage_zero.add_argument("--checkpoint", type=Path, required=True)
    persistent_stage_zero.add_argument("--cuda-library", type=Path, required=True)
    persistent_stage_zero.add_argument("--oracle-trace", type=Path, required=True)
    persistent_stage_zero.add_argument("--oracle-routes", type=Path, required=True)
    persistent_stage_zero.add_argument("--identity-manifest", type=Path, required=True)
    persistent_stage_zero.add_argument("--prior-regression", type=Path, required=True)
    persistent_stage_zero.add_argument("--output", type=Path, required=True)
    persistent_stage_zero.add_argument("--cycle-id", default="H014-026e")
    persistent_stage_profile = subparsers.add_parser(
        "persistent-stage-profile",
        help="profile canonical resident real-Kimi KDA and Gated-MLA stages",
    )
    persistent_stage_profile.add_argument("--checkpoint", type=Path, required=True)
    persistent_stage_profile.add_argument("--cuda-library", type=Path, required=True)
    persistent_stage_profile.add_argument("--oracle-trace", type=Path, required=True)
    persistent_stage_profile.add_argument("--oracle-routes", type=Path, required=True)
    persistent_stage_profile.add_argument("--identity-manifest", type=Path, required=True)
    persistent_stage_profile.add_argument("--output", type=Path, required=True)
    persistent_stage_profile.add_argument("--device", type=int, default=0)
    persistent_stage_profile.add_argument("--warmup", type=int, default=20)
    persistent_stage_profile.add_argument("--iterations", type=int, default=100)
    depth_profile = subparsers.add_parser(
        "depth-profile",
        help="profile early, middle, and late real Kimi KDA/MLA stages",
    )
    depth_profile.add_argument("--checkpoint", type=Path, required=True)
    depth_profile.add_argument("--cuda-library", type=Path, required=True)
    depth_profile.add_argument("--oracle-trace", type=Path, required=True)
    depth_profile.add_argument("--oracle-routes", type=Path, required=True)
    depth_profile.add_argument("--identity-manifest", type=Path, required=True)
    depth_profile.add_argument("--early-profile", type=Path, required=True)
    depth_profile.add_argument("--output", type=Path, required=True)
    depth_profile.add_argument("--device", type=int, default=0)
    depth_profile.add_argument("--warmup", type=int, default=20)
    depth_profile.add_argument("--iterations", type=int, default=100)
    concurrency_baseline = subparsers.add_parser(
        "concurrency-baseline",
        help="measure synchronous multi-stream service on the slowest Kimi stage",
    )
    concurrency_baseline.add_argument("--checkpoint", type=Path, required=True)
    concurrency_baseline.add_argument("--cuda-library", type=Path, required=True)
    concurrency_baseline.add_argument("--oracle-trace", type=Path, required=True)
    concurrency_baseline.add_argument("--oracle-routes", type=Path, required=True)
    concurrency_baseline.add_argument("--identity-manifest", type=Path, required=True)
    concurrency_baseline.add_argument("--depth-profile", type=Path, required=True)
    concurrency_baseline.add_argument("--output", type=Path, required=True)
    concurrency_baseline.add_argument("--device", type=int, default=0)
    concurrency_baseline.add_argument("--warmup", type=int, default=20)
    concurrency_baseline.add_argument("--iterations", type=int, default=100)
    concurrency_baseline.add_argument("--cycle-id", default="H014-027n")
    reset_contract = subparsers.add_parser(
        "reset-contract",
        help="validate bounded RESET activity across real request traces",
    )
    reset_contract.add_argument("--trace-a", type=Path, required=True)
    reset_contract.add_argument("--trace-b", type=Path, required=True)
    reset_contract.add_argument("--output", type=Path, required=True)
    batch2_primitives = subparsers.add_parser(
        "batch2-primitives",
        help="benchmark existing real-Kimi CUDA batch-2 primitives",
    )
    batch2_primitives.add_argument("--checkpoint", type=Path, required=True)
    batch2_primitives.add_argument("--cuda-library", type=Path, required=True)
    batch2_primitives.add_argument("--oracle-trace", type=Path, required=True)
    batch2_primitives.add_argument("--oracle-routes", type=Path, required=True)
    batch2_primitives.add_argument("--identity-manifest", type=Path, required=True)
    batch2_primitives.add_argument("--depth-profile", type=Path, required=True)
    batch2_primitives.add_argument("--reset-contract", type=Path, required=True)
    batch2_primitives.add_argument("--output", type=Path, required=True)
    batch2_primitives.add_argument("--device", type=int, default=0)
    batch2_primitives.add_argument("--warmup", type=int, default=20)
    batch2_primitives.add_argument("--iterations", type=int, default=100)
    batch_safety = subparsers.add_parser(
        "batch-safety",
        help="run the bounded H014-027w fail-closed post-reboot validation",
    )
    batch_safety.add_argument("--checkpoint", type=Path, required=True)
    batch_safety.add_argument("--candidate-library", type=Path, required=True)
    batch_safety.add_argument("--retained-library", type=Path, required=True)
    batch_safety.add_argument("--candidate-receipt", type=Path, required=True)
    batch_safety.add_argument("--output", type=Path, required=True)
    batch_safety.add_argument("--device", type=int, default=0)
    batch_safety.add_argument("--layer", type=int, default=89)
    batch_safety.add_argument("--expert", type=int, default=803)
    batch_safety.add_argument("--warmup", type=int, default=20)
    batch_safety.add_argument("--iterations", type=int, default=100)
    row_cooperative = subparsers.add_parser(
        "row-cooperative-batch2",
        help="benchmark only the H014-028 real-expert batch-2 reuse candidate",
    )
    row_cooperative.add_argument("--checkpoint", type=Path, required=True)
    row_cooperative.add_argument("--candidate-library", type=Path, required=True)
    row_cooperative.add_argument("--deployed-library", type=Path, required=True)
    row_cooperative.add_argument("--cuda-source", type=Path, required=True)
    row_cooperative.add_argument("--cuda-header", type=Path, required=True)
    row_cooperative.add_argument("--cuobjdump", type=Path, required=True)
    row_cooperative.add_argument("--output", type=Path, required=True)
    row_cooperative.add_argument("--device", type=int, default=0)
    row_cooperative.add_argument("--layer", type=int, default=89)
    row_cooperative.add_argument("--expert", type=int, default=803)
    row_cooperative.add_argument("--warmup", type=int, default=50)
    row_cooperative.add_argument("--iterations", type=int, default=300)
    incremental_batch = subparsers.add_parser(
        "incremental-batch",
        help="certify exactly one new H014-028 row-cooperative batch size",
    )
    incremental_batch.add_argument("--checkpoint", type=Path, required=True)
    incremental_batch.add_argument("--candidate-library", type=Path, required=True)
    incremental_batch.add_argument("--prior-library", type=Path, required=True)
    incremental_batch.add_argument("--prior-receipt", type=Path, required=True)
    incremental_batch.add_argument("--cuda-source", type=Path, required=True)
    incremental_batch.add_argument("--cuda-header", type=Path, required=True)
    incremental_batch.add_argument("--cuobjdump", type=Path, required=True)
    incremental_batch.add_argument("--output", type=Path, required=True)
    incremental_batch.add_argument("--target-batch", type=int, required=True)
    incremental_batch.add_argument("--minimum-target-speedup", type=float, required=True)
    incremental_batch.add_argument("--cycle-id", required=True)
    incremental_batch.add_argument("--device", type=int, default=0)
    incremental_batch.add_argument("--layer", type=int, default=89)
    incremental_batch.add_argument("--expert", type=int, default=803)
    incremental_batch.add_argument("--warmup", type=int, default=50)
    incremental_batch.add_argument("--iterations", type=int, default=300)
    complete_stage_batch = subparsers.add_parser(
        "complete-stage-batch",
        help="incrementally certify real-route batching on one complete Kimi stage",
    )
    complete_stage_batch.add_argument("--checkpoint", type=Path, required=True)
    complete_stage_batch.add_argument("--cuda-library", type=Path, required=True)
    complete_stage_batch.add_argument("--oracle-trace", type=Path, required=True)
    complete_stage_batch.add_argument("--oracle-routes", type=Path, required=True)
    complete_stage_batch.add_argument("--identity-manifest", type=Path, required=True)
    complete_stage_batch.add_argument("--graph-certification", type=Path, required=True)
    complete_stage_batch.add_argument("--output", type=Path, required=True)
    complete_stage_batch.add_argument("--layer", type=int, required=True)
    complete_stage_batch.add_argument("--target-batch", type=int, default=16)
    complete_stage_batch.add_argument("--cycle-id", required=True)
    complete_stage_batch.add_argument("--device", type=int, default=0)
    complete_stage_batch.add_argument("--warmup", type=int, default=10)
    complete_stage_batch.add_argument("--iterations", type=int, default=50)
    complete_stage_batch.add_argument("--phase-iterations", type=int, default=5)
    continuous_batch = subparsers.add_parser(
        "continuous-batch",
        help="certify position-aware continuous batching on one real Kimi stage",
    )
    continuous_batch.add_argument("--checkpoint", type=Path, required=True)
    continuous_batch.add_argument("--cuda-library", type=Path, required=True)
    continuous_batch.add_argument("--oracle-trace", type=Path, required=True)
    continuous_batch.add_argument("--oracle-routes", type=Path, required=True)
    continuous_batch.add_argument("--identity-manifest", type=Path, required=True)
    continuous_batch.add_argument("--graph-certification", type=Path, required=True)
    continuous_batch.add_argument("--output", type=Path, required=True)
    continuous_batch.add_argument("--layer", type=int, default=89)
    continuous_batch.add_argument("--batch", type=int, default=8)
    continuous_batch.add_argument("--cycle-id", default="H014-031a")
    continuous_batch.add_argument("--device", type=int, default=0)
    continuous_batch.add_argument("--warmup", type=int, default=10)
    continuous_batch.add_argument("--iterations", type=int, default=50)
    contextual_batch = subparsers.add_parser(
        "contextual-batch",
        help="incrementally certify complete-stage batching at a populated MLA cache",
    )
    contextual_batch.add_argument("--checkpoint", type=Path, required=True)
    contextual_batch.add_argument("--cuda-library", type=Path, required=True)
    contextual_batch.add_argument("--oracle-trace", type=Path, required=True)
    contextual_batch.add_argument("--reference-receipt", type=Path, required=True)
    contextual_batch.add_argument("--long-context-receipt", type=Path, required=True)
    contextual_batch.add_argument("--output", type=Path, required=True)
    contextual_batch.add_argument("--layer", type=int, default=91)
    contextual_batch.add_argument("--context", type=int, default=8192)
    contextual_batch.add_argument("--cycle-id", default="H014-034a")
    contextual_batch.add_argument("--device", type=int, default=0)
    contextual_batch.add_argument("--warmup", type=int, default=5)
    contextual_batch.add_argument("--iterations", type=int, default=20)
    contextual_continuous = subparsers.add_parser(
        "contextual-continuous",
        help="measure eight continuous real Kimi streams at an 8K MLA cache",
    )
    contextual_continuous.add_argument("--checkpoint", type=Path, required=True)
    contextual_continuous.add_argument("--cuda-library", type=Path, required=True)
    contextual_continuous.add_argument("--oracle-trace", type=Path, required=True)
    contextual_continuous.add_argument("--oracle-routes", type=Path, required=True)
    contextual_continuous.add_argument("--identity-manifest", type=Path, required=True)
    contextual_continuous.add_argument("--graph-certification", type=Path, required=True)
    contextual_continuous.add_argument("--reference-receipt", type=Path, required=True)
    contextual_continuous.add_argument("--long-context-receipt", type=Path, required=True)
    contextual_continuous.add_argument("--static-contextual-receipt", type=Path, required=True)
    contextual_continuous.add_argument("--prior-continuous-receipt", type=Path)
    contextual_continuous.add_argument("--output", type=Path, required=True)
    contextual_continuous.add_argument("--layer", type=int, default=91)
    contextual_continuous.add_argument("--context", type=int, default=8192)
    contextual_continuous.add_argument("--batch", type=int, default=8)
    contextual_continuous.add_argument("--cycle-id", default="H014-034e")
    contextual_continuous.add_argument("--device", type=int, default=0)
    contextual_continuous.add_argument("--warmup", type=int, default=5)
    contextual_continuous.add_argument("--iterations", type=int, default=50)
    coarse_stage_tcp = subparsers.add_parser(
        "coarse-stage-tcp",
        help="certify a real persistent stage-0 to stage-1 Kimi CUDA/TCP slice",
    )
    coarse_stage_tcp.add_argument("--checkpoint", type=Path, required=True)
    coarse_stage_tcp.add_argument("--cuda-library", type=Path, required=True)
    coarse_stage_tcp.add_argument("--oracle-trace", type=Path, required=True)
    coarse_stage_tcp.add_argument("--oracle-routes", type=Path, required=True)
    coarse_stage_tcp.add_argument("--graph-certification", type=Path, required=True)
    coarse_stage_tcp.add_argument("--output", type=Path, required=True)
    coarse_stage_tcp.add_argument("--cycle-id", default="H014-032a")
    coarse_stage_tcp.add_argument("--device", type=int, default=0)
    coarse_stage_tcp.add_argument("--warmup", type=int, default=10)
    coarse_stage_tcp.add_argument("--iterations", type=int, default=50)
    coarse_stage_tcp.add_argument("--ready-timeout-seconds", type=float, default=240.0)
    coarse_network = subparsers.add_parser(
        "coarse-network-analysis",
        help="replay the measured Kimi coarse edge under RTT and bandwidth profiles",
    )
    coarse_network.add_argument("--coarse-receipt", type=Path, required=True)
    coarse_network.add_argument("--fine-receipt", type=Path, required=True)
    coarse_network.add_argument("--output", type=Path, required=True)
    coarse_network.add_argument("--csv", type=Path, required=True)
    coarse_network.add_argument("--chart", type=Path, required=True)
    coarse_network.add_argument("--edge-class", type=Path, required=True)
    coarse_network.add_argument("--cycle-id", default="H014-032d")
    boundary_formats = subparsers.add_parser(
        "coarse-boundary-formats",
        help="certify FP32, FP16 and BF16 real Kimi coarse boundaries",
    )
    boundary_formats.add_argument("--checkpoint", type=Path, required=True)
    boundary_formats.add_argument("--cuda-library", type=Path, required=True)
    boundary_formats.add_argument("--oracle-trace", type=Path, required=True)
    boundary_formats.add_argument("--oracle-routes", type=Path, required=True)
    boundary_formats.add_argument("--graph-certification", type=Path, required=True)
    boundary_formats.add_argument("--coarse-receipt", type=Path, required=True)
    boundary_formats.add_argument("--output", type=Path, required=True)
    boundary_formats.add_argument("--device", type=int, default=0)
    boundary_formats.add_argument("--timing-iterations", type=int, default=100)
    boundary_formats.add_argument("--cycle-id", default="H014-032e")
    batched_router = subparsers.add_parser(
        "batched-router",
        help="incrementally certify the real-weight row-cooperative Kimi router",
    )
    batched_router.add_argument("--checkpoint", type=Path, required=True)
    batched_router.add_argument("--candidate-library", type=Path, required=True)
    batched_router.add_argument("--prior-library", type=Path, required=True)
    batched_router.add_argument("--oracle-trace", type=Path, required=True)
    batched_router.add_argument("--output", type=Path, required=True)
    batched_router.add_argument("--layer", type=int, default=89)
    batched_router.add_argument("--device", type=int, default=0)
    batched_router.add_argument("--warmup", type=int, default=30)
    batched_router.add_argument("--iterations", type=int, default=200)
    batched_router.add_argument("--cycle-id", default="H014-030a")
    dense_row_reuse = subparsers.add_parser(
        "dense-row-reuse",
        help="incrementally benchmark real-weight row-reuse dense projection",
    )
    dense_row_reuse.add_argument("--checkpoint", type=Path, required=True)
    dense_row_reuse.add_argument("--cuda-library", type=Path, required=True)
    dense_row_reuse.add_argument("--oracle-trace", type=Path, required=True)
    dense_row_reuse.add_argument("--output", type=Path, required=True)
    dense_row_reuse.add_argument("--layer", type=int, default=89)
    dense_row_reuse.add_argument("--role", default="q")
    dense_row_reuse.add_argument("--device", type=int, default=0)
    dense_row_reuse.add_argument("--warmup", type=int, default=30)
    dense_row_reuse.add_argument("--iterations", type=int, default=200)
    dense_row_reuse.add_argument("--cycle-id", default="H014-030d")
    expert_microwork = subparsers.add_parser(
        "expert-microwork",
        help="certify real Kimi routed experts across persistent worker partitions",
    )
    expert_microwork.add_argument("--checkpoint", type=Path, required=True)
    expert_microwork.add_argument("--cuda-library", type=Path, required=True)
    expert_microwork.add_argument("--oracle-trace", type=Path, required=True)
    expert_microwork.add_argument("--oracle-routes", type=Path, required=True)
    expert_microwork.add_argument("--graph-certification", type=Path, required=True)
    expert_microwork.add_argument("--output", type=Path, required=True)
    expert_microwork.add_argument("--layer", type=int, default=89)
    expert_microwork.add_argument("--workers", type=int, required=True)
    expert_microwork.add_argument("--device", type=int, default=0)
    expert_microwork.add_argument("--warmup", type=int, default=10)
    expert_microwork.add_argument("--iterations", type=int, default=30)
    expert_microwork.add_argument("--cycle-id", required=True)
    expert_microwork.add_argument("--maximum-critical-worker-device-ms", type=float)
    expert_microwork.add_argument("--minimum-relative-throughput", type=float)
    sub_layer_analysis = subparsers.add_parser(
        "sub-layer-analysis",
        help="build the real microwork scaling curve and physical-link replay",
    )
    sub_layer_analysis.add_argument("--receipt-2", type=Path, required=True)
    sub_layer_analysis.add_argument("--receipt-4", type=Path, required=True)
    sub_layer_analysis.add_argument("--receipt-8", type=Path, required=True)
    sub_layer_analysis.add_argument("--receipt-16", type=Path, required=True)
    sub_layer_analysis.add_argument("--output", type=Path, required=True)
    sub_layer_analysis.add_argument("--csv", type=Path, required=True)
    sub_layer_analysis.add_argument("--chart", type=Path, required=True)
    sub_layer_analysis.add_argument("--cycle-id", default="H014-SUB-005")
    sub_layer_batch = subparsers.add_parser(
        "sub-layer-batch",
        help="incrementally certify batch 1/2/4/8 on the real expert collective",
    )
    sub_layer_batch.add_argument("--checkpoint", type=Path, required=True)
    sub_layer_batch.add_argument("--cuda-library", type=Path, required=True)
    sub_layer_batch.add_argument("--oracle-trace", type=Path, required=True)
    sub_layer_batch.add_argument("--graph-certification", type=Path, required=True)
    sub_layer_batch.add_argument("--output", type=Path, required=True)
    sub_layer_batch.add_argument("--layer", type=int, default=89)
    sub_layer_batch.add_argument("--workers", type=int, default=4)
    sub_layer_batch.add_argument("--device", type=int, default=0)
    sub_layer_batch.add_argument("--warmup", type=int, default=5)
    sub_layer_batch.add_argument("--iterations", type=int, default=20)
    sub_layer_batch.add_argument("--cycle-id", default="H014-SUB-006")
    sub_layer_batch.add_argument("--overlap-parent-shared", action="store_true")
    shared_expert_placement = subparsers.add_parser(
        "shared-expert-placement",
        help="join real bytes/timing to select parent or remote shared experts",
    )
    shared_expert_placement.add_argument("--checkpoint", type=Path, required=True)
    shared_expert_placement.add_argument("--complete-batch-receipt", type=Path, required=True)
    shared_expert_placement.add_argument("--distributed-batch-receipt", type=Path, required=True)
    shared_expert_placement.add_argument("--fine-network-receipt", type=Path, required=True)
    shared_expert_placement.add_argument("--output", type=Path, required=True)
    shared_expert_placement.add_argument("--cycle-id", default="H014-SUB-007")
    sub_layer_recovery = subparsers.add_parser(
        "sub-layer-recovery",
        help="certify fail-closed recovery of the real expert collective",
    )
    sub_layer_recovery.add_argument("--checkpoint", type=Path, required=True)
    sub_layer_recovery.add_argument("--cuda-library", type=Path, required=True)
    sub_layer_recovery.add_argument("--oracle-trace", type=Path, required=True)
    sub_layer_recovery.add_argument("--passing-batch-receipt", type=Path, required=True)
    sub_layer_recovery.add_argument("--output", type=Path, required=True)
    sub_layer_recovery.add_argument("--device", type=int, default=0)
    sub_layer_recovery.add_argument("--cycle-id", default="H014-SUB-009")
    sub_layer_routing = subparsers.add_parser(
        "sub-layer-routing",
        help="measure real-route imbalance and held-out expert ownership",
    )
    sub_layer_routing.add_argument("--oracle-routes", type=Path, required=True)
    sub_layer_routing.add_argument("--serial-receipt", type=Path, required=True)
    sub_layer_routing.add_argument("--recovery-receipt", type=Path, required=True)
    sub_layer_routing.add_argument("--output", type=Path, required=True)
    sub_layer_routing.add_argument("--csv", type=Path, required=True)
    sub_layer_routing.add_argument("--workers", type=int, default=4)
    sub_layer_routing.add_argument("--cycle-id", default="H014-SUB-010")
    prefill_stage = subparsers.add_parser(
        "prefill-stage",
        help="incrementally certify one real Kimi stage at 1K/4K/8K/16K",
    )
    prefill_stage.add_argument("--checkpoint", type=Path, required=True)
    prefill_stage.add_argument("--cuda-library", type=Path, required=True)
    prefill_stage.add_argument("--oracle-trace", type=Path, required=True)
    prefill_stage.add_argument("--reference-receipt", type=Path, required=True)
    prefill_stage.add_argument("--output", type=Path, required=True)
    prefill_stage.add_argument("--layer", type=int, required=True)
    prefill_stage.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    prefill_stage.add_argument("--device", type=int, default=0)
    prefill_stage.add_argument("--cycle-id", default="H014-033a")
    prefill_stage.add_argument("--terminal-ratio-floor", type=float, default=0.0)
    prefill_stage.add_argument("--terminal-ratio-gate", type=float, default=1.10)
    prefill_stage.add_argument("--throughput-retention-gate", type=float, default=0.90)
    prefill_stage.add_argument(
        "--candidate-regression",
        action="store_true",
        help="permit a new CUDA SHA while retaining exact reference gates",
    )
    model_preregister = subparsers.add_parser(
        "model-preregister",
        help="freeze P4 held-out timing predictions before executing them",
    )
    model_preregister.add_argument("--depth-profile", type=Path, required=True)
    model_preregister.add_argument("--mla-context-profile", type=Path, required=True)
    model_preregister.add_argument("--cuda-library", type=Path, required=True)
    model_preregister.add_argument("--output", type=Path, required=True)
    model_preregister.add_argument("--cycle-id", default="H014-035a")
    model_preregister.add_argument("--median-ape-gate-percent", type=float, default=10.0)
    model_validate = subparsers.add_parser(
        "model-validate",
        help="execute the frozen P4 real-Kimi held-out slices without refitting",
    )
    model_validate.add_argument("--checkpoint", type=Path, required=True)
    model_validate.add_argument("--cuda-library", type=Path, required=True)
    model_validate.add_argument("--oracle-trace", type=Path, required=True)
    model_validate.add_argument("--oracle-routes", type=Path, required=True)
    model_validate.add_argument("--identity-manifest", type=Path, required=True)
    model_validate.add_argument("--reference-receipt", type=Path, required=True)
    model_validate.add_argument("--preregistration", type=Path, required=True)
    model_validate.add_argument("--output", type=Path, required=True)
    model_validate.add_argument("--device", type=int, default=0)
    model_validate.add_argument("--warmup", type=int, default=10)
    model_validate.add_argument("--iterations", type=int, default=50)
    model_validate.add_argument("--cycle-id", default="H014-035b")
    capacity_topology = subparsers.add_parser(
        "capacity-topology",
        help="build the source-backed P4/P5/P6 capacity and topology decision",
    )
    for source_name in (
        "validation",
        "resident-profile",
        "depth-profile",
        "stage-zero",
        "final-stage",
        "lm-head",
        "complete-kda-batch",
        "complete-mla-batch",
        "contextual-static",
        "contextual-route-mix",
        "contextual-matched",
        "kda-prefill",
        "mla-prefill",
        "sub-layer-scaling",
        "sub-layer-batch",
        "coarse-transport",
        "coarse-network",
        "node-solver",
    ):
        capacity_topology.add_argument(f"--{source_name}", type=Path, required=True)
    capacity_topology.add_argument("--output", type=Path, required=True)
    capacity_topology.add_argument("--candidate-csv", type=Path, required=True)
    capacity_topology.add_argument("--economics-csv", type=Path, required=True)
    capacity_topology.add_argument("--chart", type=Path, required=True)
    capacity_topology.add_argument("--cycle-id", default="H014-036a")
    final_placement = subparsers.add_parser(
        "final-placement",
        help="merge exact ownership into the selected 93-worker production topology",
    )
    final_placement.add_argument("--old-placement", type=Path, required=True)
    final_placement.add_argument("--capacity-receipt", type=Path, required=True)
    final_placement.add_argument("--stage-zero-receipt", type=Path, required=True)
    final_placement.add_argument("--kda-batch-receipt", type=Path, required=True)
    final_placement.add_argument("--mla-batch-receipt", type=Path, required=True)
    final_placement.add_argument("--final-stage-receipt", type=Path, required=True)
    final_placement.add_argument("--cuda-library", type=Path, required=True)
    final_placement.add_argument("--promotion-receipt", type=Path, required=True)
    final_placement.add_argument("--output", type=Path, required=True)
    final_placement.add_argument("--cycle-id", default="H014-037a")
    acquisition_fixture = subparsers.add_parser(
        "acquisition-fixture",
        help="test resumable immutable worker acquisition and atomic activation",
    )
    acquisition_fixture.add_argument("--output", type=Path, required=True)
    acquisition_fixture.add_argument("--cycle-id", default="H014-037a")
    tokenizer_product_fixture = subparsers.add_parser(
        "tokenizer-product-fixture",
        help="certify the exact-hash stage-zero Kimi tokenizer seam",
    )
    tokenizer_product_fixture.add_argument(
        "--tokenizer-directory", type=Path, required=True
    )
    tokenizer_product_fixture.add_argument(
        "--conversation-receipt", type=Path, required=True
    )
    tokenizer_product_fixture.add_argument("--output", type=Path, required=True)
    tokenizer_product_fixture.add_argument("--cycle-id", default="H014-037b2a")
    acquire_worker = subparsers.add_parser(
        "acquire-worker",
        help="download and atomically activate one assigned Kimi worker package",
    )
    acquire_worker.add_argument("--distribution-manifest", type=Path, required=True)
    acquire_worker.add_argument("--worker-id", required=True)
    acquire_worker.add_argument("--cache-directory", type=Path, required=True)
    acquire_worker.add_argument("--output", type=Path, required=True)
    acquire_worker.add_argument("--retries", type=int, default=4)
    acquire_worker.add_argument("--timeout-seconds", type=float, default=30.0)
    activate_snapshot = subparsers.add_parser(
        "activate-worker-snapshot",
        help="atomically activate a directly loadable worker-only Kimi snapshot",
    )
    activate_snapshot.add_argument("--placement", type=Path, required=True)
    activate_snapshot.add_argument("--worker-id", required=True)
    activate_snapshot.add_argument("--package", type=Path, required=True)
    activate_snapshot.add_argument("--config", type=Path, required=True)
    activate_snapshot.add_argument("--model-id", default="moonshotai/Kimi-K3")
    activate_snapshot.add_argument("--tokenizer-revision")
    activate_snapshot.add_argument("--adapter-id", default="kimi_k3_cuda")
    activate_snapshot.add_argument("--output-directory", type=Path, required=True)
    deployment_recovery = subparsers.add_parser(
        "deployment-recovery",
        help="inject fail-closed coarse-stage recovery faults",
    )
    deployment_recovery.add_argument("--output", type=Path, required=True)
    deployment_recovery.add_argument("--cycle-id", default="H014-037b")
    admission_fixture = subparsers.add_parser(
        "admission-fixture",
        help="test fail-closed node admission and the fleet cost guard",
    )
    admission_fixture.add_argument("--output", type=Path, required=True)
    admission_fixture.add_argument("--cycle-id", default="H014-037b")
    node_admission = subparsers.add_parser(
        "node-admission", help="evaluate a measured Experiment 015 worker"
    )
    node_admission.add_argument("--observation", type=Path, required=True)
    node_admission.add_argument("--requirements", type=Path, required=True)
    node_admission.add_argument("--output", type=Path, required=True)
    node_admission.add_argument(
        "--mode", choices=("PRE_CANARY", "FLEET"), required=True
    )
    fleet_cost_guard = subparsers.add_parser(
        "fleet-cost-guard", help="fail closed on topology, health, or cost"
    )
    fleet_cost_guard.add_argument("--fleet-status", type=Path, required=True)
    fleet_cost_guard.add_argument("--policy", type=Path, required=True)
    fleet_cost_guard.add_argument("--output", type=Path, required=True)
    worker_requirements = subparsers.add_parser(
        "worker-requirements", help="derive one exact worker admission contract"
    )
    worker_requirements.add_argument("--placement", type=Path, required=True)
    worker_requirements.add_argument("--distribution", type=Path, required=True)
    worker_requirements.add_argument("--worker-id", required=True)
    worker_requirements.add_argument("--package-version", required=True)
    worker_requirements.add_argument("--runtime-sha256", default="")
    worker_requirements.add_argument("--output", type=Path, required=True)
    inspect_deployment_node = subparsers.add_parser(
        "inspect-deployment-node",
        help="inspect hardware before CUDA and retain measured network input",
    )
    inspect_deployment_node.add_argument("--runtime", type=Path)
    inspect_deployment_node.add_argument("--assigned-stage-canary", type=Path)
    inspect_deployment_node.add_argument("--network-rtt-ms", type=float, required=True)
    inspect_deployment_node.add_argument(
        "--network-bandwidth-gbps", type=float, required=True
    )
    inspect_deployment_node.add_argument("--assignment-sha256", required=True)
    inspect_deployment_node.add_argument("--checkpoint-revision", required=True)
    inspect_deployment_node.add_argument("--package-version", required=True)
    inspect_deployment_node.add_argument("--disk-path", type=Path, required=True)
    inspect_deployment_node.add_argument("--device", type=int, default=0)
    inspect_deployment_node.add_argument("--output", type=Path, required=True)
    bind_deployment_plan = subparsers.add_parser(
        "bind-deployment-plan",
        help="bind the certified placement to exactly 93 healthy live workers",
    )
    bind_deployment_plan.add_argument("--placement", type=Path, required=True)
    bind_deployment_plan.add_argument("--workers-status", type=Path, required=True)
    bind_deployment_plan.add_argument(
        "--runtime-certificate", type=Path, required=True
    )
    bind_deployment_plan.add_argument(
        "--native-source-manifest", type=Path, required=True
    )
    bind_deployment_plan.add_argument("--output", type=Path, required=True)
    deployment_identity_fixture = subparsers.add_parser(
        "deployment-identity-fixture",
        help="certify fail-closed Kimi model/runtime identity propagation",
    )
    deployment_identity_fixture.add_argument("--placement", type=Path, required=True)
    deployment_identity_fixture.add_argument("--output", type=Path, required=True)
    deployment_identity_fixture.add_argument("--cycle-id", default="H014-037b1")
    canary_fixtures = subparsers.add_parser(
        "build-physical-canary-fixtures",
        help="distil exact Kimi oracle inputs for the Experiment 015 3090 canary",
    )
    canary_fixtures.add_argument("--checkpoint", type=Path, required=True)
    canary_fixtures.add_argument("--oracle-trace", type=Path, required=True)
    canary_fixtures.add_argument("--oracle-routes", type=Path, required=True)
    canary_fixtures.add_argument("--oracle-logits", type=Path, required=True)
    canary_fixtures.add_argument("--placement", type=Path, required=True)
    canary_fixtures.add_argument("--full-graph-receipt", type=Path, required=True)
    canary_fixtures.add_argument("--operation-matrix-receipt", type=Path, required=True)
    canary_fixtures.add_argument("--sm86-receipt", type=Path, required=True)
    canary_fixtures.add_argument("--output-npz", type=Path, required=True)
    canary_fixtures.add_argument("--output-manifest", type=Path, required=True)
    physical_canary = subparsers.add_parser(
        "physical-3090-canary",
        help="qualify the canary-built Linux Kimi runtime on one physical RTX 3090",
    )
    physical_canary.add_argument("--placement", type=Path, required=True)
    physical_canary.add_argument("--native-source-manifest", type=Path, required=True)
    physical_canary.add_argument("--runtime", type=Path, required=True)
    physical_canary.add_argument("--fixture-npz", type=Path, required=True)
    physical_canary.add_argument("--fixture-manifest", type=Path, required=True)
    physical_canary.add_argument("--evidence-root", type=Path, required=True)
    physical_canary.add_argument("--stage-zero-snapshot", type=Path, required=True)
    physical_canary.add_argument("--kda-snapshot", type=Path, required=True)
    physical_canary.add_argument("--mla-snapshot", type=Path, required=True)
    physical_canary.add_argument("--final-snapshot", type=Path, required=True)
    physical_canary.add_argument("--receipt", type=Path, required=True)
    physical_canary.add_argument("--certificate", type=Path, required=True)
    assigned_canary = subparsers.add_parser(
        "assigned-stage-canary",
        help="qualify one exact real Kimi worker assignment before registration",
    )
    assigned_canary.add_argument("--placement", type=Path, required=True)
    assigned_canary.add_argument("--native-source-manifest", type=Path, required=True)
    assigned_canary.add_argument("--runtime-certificate", type=Path, required=True)
    assigned_canary.add_argument("--runtime", type=Path, required=True)
    assigned_canary.add_argument("--snapshot", type=Path, required=True)
    assigned_canary.add_argument("--worker-id", required=True)
    assigned_canary.add_argument("--output", type=Path, required=True)
    deployment_bundle = subparsers.add_parser(
        "build-experiment-015-package",
        help="build the hash-locked no-checkout Experiment 015 deployment package",
    )
    deployment_bundle.add_argument("--repository-root", type=Path, required=True)
    deployment_bundle.add_argument("--output", type=Path, required=True)
    deployment_bundle.add_argument("--wheel", type=Path, required=True)
    deployment_bundle.add_argument(
        "--checkpoint-metadata-directory", type=Path, required=True
    )
    deployment_validation = subparsers.add_parser(
        "validate-experiment-015-package",
        help="verify Experiment 015 package closure and fail-closed launch seams",
    )
    deployment_validation.add_argument("--package", type=Path, required=True)
    deployment_controls = subparsers.add_parser(
        "experiment-015-package-controls",
        help="run the bounded Experiment 015 package tamper controls",
    )
    deployment_controls.add_argument("--package", type=Path, required=True)
    deployment_controls.add_argument("--output", type=Path, required=True)
    deployment_controls.add_argument("--cycle-id", default="H014-037b4")
    batch_scaling = subparsers.add_parser(
        "batch-scaling",
        help="measure real-Kimi CUDA primitive scaling at batch 1/2/4/8/16",
    )
    batch_scaling.add_argument("--checkpoint", type=Path, required=True)
    batch_scaling.add_argument("--cuda-library", type=Path, required=True)
    batch_scaling.add_argument("--oracle-trace", type=Path, required=True)
    batch_scaling.add_argument("--oracle-routes", type=Path, required=True)
    batch_scaling.add_argument("--identity-manifest", type=Path, required=True)
    batch_scaling.add_argument("--batch2-profile", type=Path, required=True)
    batch_scaling.add_argument("--output", type=Path, required=True)
    batch_scaling.add_argument("--device", type=int, default=0)
    batch_scaling.add_argument("--warmup", type=int, default=20)
    batch_scaling.add_argument("--iterations", type=int, default=100)
    expert_working_set = subparsers.add_parser(
        "expert-working-set",
        help="compare fixed and rotating resident Kimi expert working sets",
    )
    expert_working_set.add_argument("--checkpoint", type=Path, required=True)
    expert_working_set.add_argument("--cuda-library", type=Path, required=True)
    expert_working_set.add_argument("--oracle-trace", type=Path, required=True)
    expert_working_set.add_argument("--oracle-routes", type=Path, required=True)
    expert_working_set.add_argument("--identity-manifest", type=Path, required=True)
    expert_working_set.add_argument("--steady-profile", type=Path, required=True)
    expert_working_set.add_argument("--output", type=Path, required=True)
    expert_working_set.add_argument("--device", type=int, default=0)
    device_warmup = subparsers.add_parser(
        "device-warmup",
        help="test unrelated CUDA compute warmup before the first Kimi stage call",
    )
    device_warmup.add_argument("--checkpoint", type=Path, required=True)
    device_warmup.add_argument("--cuda-library", type=Path, required=True)
    device_warmup.add_argument("--oracle-trace", type=Path, required=True)
    device_warmup.add_argument("--oracle-routes", type=Path, required=True)
    device_warmup.add_argument("--identity-manifest", type=Path, required=True)
    device_warmup.add_argument("--steady-profile", type=Path, required=True)
    device_warmup.add_argument("--cold-profile", type=Path, required=True)
    device_warmup.add_argument("--output", type=Path, required=True)
    device_warmup.add_argument("--device", type=int, default=0)
    device_warmup.add_argument("--minimum-warmup-device-ms", type=float, default=100.0)
    integrated_readiness = subparsers.add_parser(
        "integrated-readiness",
        help="validate production Kimi PREPARE warmup before READY",
    )
    integrated_readiness.add_argument("--checkpoint", type=Path, required=True)
    integrated_readiness.add_argument("--cuda-library", type=Path, required=True)
    integrated_readiness.add_argument("--oracle-trace", type=Path, required=True)
    integrated_readiness.add_argument("--oracle-routes", type=Path, required=True)
    integrated_readiness.add_argument("--identity-manifest", type=Path, required=True)
    integrated_readiness.add_argument("--steady-profile", type=Path, required=True)
    integrated_readiness.add_argument("--output", type=Path, required=True)
    integrated_readiness.add_argument("--device", type=int, default=0)
    integrated_readiness.add_argument("--cycle-id", default="H014-027d")
    distribution = subparsers.add_parser(
        "distribution", help="build exact worker model distribution"
    )
    distribution.add_argument("--checkpoint", type=Path, required=True)
    distribution.add_argument("--placement", type=Path, required=True)
    distribution.add_argument("--output", type=Path, required=True)
    materialize = subparsers.add_parser(
        "materialize-worker", help="extract one exact worker package"
    )
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
            k3_idot=args.k3_idot,
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
    if args.command == "cuda-operation-matrix":
        receipt = build_cuda_operation_matrix(
            args.kimi_source,
            args.cuda_source,
            args.makefile,
            args.output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "cuda-expert":
        receipt = benchmark_real_mxfp4_expert(
            args.checkpoint,
            args.cuda_library,
            args.reference_library,
            args.output,
            device=args.device,
            layer=args.layer,
            expert=args.expert,
            warmup=args.warmup,
            iterations=args.iterations,
            batch=args.batch,
            seed=args.seed,
            cuda_architecture=args.cuda_architecture,
            up_first=args.up_first,
            cycle_id=args.cycle_id,
            fuse_gate_up=args.fuse_gate_up,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "sm86-certification":
        receipt = certify_sm86_routed_expert(
            args.operation_matrix,
            args.benchmark,
            args.binary,
            args.cuobjdump,
            args.output,
            router_benchmark_path=args.router_benchmark,
            dense_benchmark_paths=args.dense_benchmark,
            final_norm_benchmark_path=args.final_norm_benchmark,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "sm86-full-certification":
        receipt = certify_sm86_complete_kimi(
            args.operation_matrix,
            args.benchmark,
            args.binary,
            args.cuobjdump,
            args.qualification,
            args.output,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-router":
        receipt = benchmark_real_router(
            args.checkpoint,
            args.cuda_library,
            args.activation,
            args.output,
            device=args.device,
            layer=args.layer,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-dense":
        receipt = benchmark_real_dense_projection(
            args.checkpoint,
            args.cuda_library,
            args.activation,
            args.output,
            device=args.device,
            layer=args.layer,
            tensor_suffix=args.tensor_suffix,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-final-norm":
        receipt = benchmark_real_final_norm(
            args.checkpoint,
            args.cuda_library,
            args.activation,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-embedding":
        receipt = benchmark_real_embedding(
            args.checkpoint,
            args.cuda_library,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-lm-head":
        receipt = benchmark_real_lm_head(
            args.checkpoint,
            args.cuda_library,
            args.trace,
            args.oracle_logits,
            args.output,
            trace_row=args.trace_row,
            logits_row=args.logits_row,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-shared-expert":
        receipt = benchmark_real_shared_expert(
            args.checkpoint,
            args.cuda_library,
            args.activation,
            args.output,
            layer=args.layer,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-moe-reduction":
        receipt = benchmark_real_moe_reduction(
            args.checkpoint,
            args.cuda_library,
            args.reference_library,
            args.activation,
            args.output,
            layer=args.layer,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-attnres":
        receipt = benchmark_real_attnres(
            args.checkpoint,
            args.cuda_library,
            args.trace,
            args.output,
            trace_step=args.trace_step,
            token_id=args.token_id,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
            cycle_id=args.cycle_id,
            baseline_path=args.baseline,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-kda-core":
        receipt = benchmark_real_kda_core(
            args.checkpoint,
            args.cuda_library,
            args.activation,
            args.trace,
            args.output,
            layer=args.layer,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-kda-stage":
        receipt = benchmark_real_kda_stage(
            args.checkpoint,
            args.cuda_library,
            args.activation,
            args.trace,
            args.output,
            layer=args.layer,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-mla-stage":
        receipt = benchmark_real_mla_stage(
            args.checkpoint,
            args.cuda_library,
            args.trace,
            args.output,
            layer=args.layer,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cuda_architecture=args.cuda_architecture,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "cuda-streamed-graph":
        receipt = benchmark_streamed_cuda_graph(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.output,
            oracle_logits_path=args.oracle_logits,
            layer_limit=args.layer_limit,
            prompt_token_ids=tuple(args.prompt_token_id or (163584, 18699)),
            decode_token_id=args.decode_token_id,
            device=args.device,
            relative_error_gate=args.relative_error_gate,
            cycle_id=args.cycle_id,
            oracle_layer_count=args.oracle_layer_count,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "persistent-final-stage":
        from swarm_inference.experiments.experiment_014.persistent_cuda import (
            benchmark_persistent_final_stage,
        )

        receipt = benchmark_persistent_final_stage(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.oracle_logits,
            args.output,
            registered=args.registered,
            identity_manifest=args.identity_manifest,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "persistent-nonfinal-stages":
        from swarm_inference.experiments.experiment_014.persistent_stages import (
            benchmark_persistent_nonfinal_stages,
        )

        receipt = benchmark_persistent_nonfinal_stages(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.final_regression,
            args.output,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "persistent-stage-zero":
        from swarm_inference.experiments.experiment_014.persistent_stages import (
            benchmark_persistent_stage_zero,
        )

        receipt = benchmark_persistent_stage_zero(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.prior_regression,
            args.output,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "persistent-stage-profile":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_resident_stage_profile,
        )

        receipt = benchmark_resident_stage_profile(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "depth-profile":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_depth_profile,
        )

        receipt = benchmark_depth_profile(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.early_profile,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "concurrency-baseline":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_concurrency_baseline,
        )

        receipt = benchmark_concurrency_baseline(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.depth_profile,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "reset-contract":
        from swarm_inference.experiments.experiment_014.performance import (
            validate_reset_contract,
        )

        receipt = validate_reset_contract(
            args.trace_a,
            args.trace_b,
            args.output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "batch2-primitives":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_batch2_primitives,
        )

        receipt = benchmark_batch2_primitives(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.depth_profile,
            args.reset_contract,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "batch-safety":
        from swarm_inference.experiments.experiment_014.batch_safety import (
            validate_failclosed_batch_candidate,
        )

        receipt = validate_failclosed_batch_candidate(
            args.checkpoint,
            args.candidate_library,
            args.retained_library,
            args.candidate_receipt,
            args.output,
            device=args.device,
            layer=args.layer,
            expert_id=args.expert,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "row-cooperative-batch2":
        from swarm_inference.experiments.experiment_014.row_cooperative import (
            benchmark_row_cooperative_batch2,
        )

        receipt = benchmark_row_cooperative_batch2(
            args.checkpoint,
            args.candidate_library,
            args.deployed_library,
            args.cuda_source,
            args.cuda_header,
            args.cuobjdump,
            args.output,
            device=args.device,
            layer=args.layer,
            expert_id=args.expert,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "incremental-batch":
        from swarm_inference.experiments.experiment_014.incremental_batch import (
            benchmark_incremental_row_cooperative_batch,
        )

        receipt = benchmark_incremental_row_cooperative_batch(
            args.checkpoint,
            args.candidate_library,
            args.prior_library,
            args.prior_receipt,
            args.cuda_source,
            args.cuda_header,
            args.cuobjdump,
            args.output,
            target_batch=args.target_batch,
            minimum_target_speedup=args.minimum_target_speedup,
            cycle_id=args.cycle_id,
            device=args.device,
            layer=args.layer,
            expert_id=args.expert,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "complete-stage-batch":
        from swarm_inference.experiments.experiment_014.complete_stage_batch import (
            benchmark_complete_stage_batch,
        )

        receipt = benchmark_complete_stage_batch(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.graph_certification,
            args.output,
            layer=args.layer,
            target_batch=args.target_batch,
            cycle_id=args.cycle_id,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            phase_iterations=args.phase_iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "continuous-batch":
        from swarm_inference.experiments.experiment_014.continuous_batch import (
            benchmark_continuous_batch,
        )

        receipt = benchmark_continuous_batch(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.graph_certification,
            args.output,
            layer=args.layer,
            batch=args.batch,
            cycle_id=args.cycle_id,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "contextual-batch":
        from swarm_inference.experiments.experiment_014.contextual_batch import (
            benchmark_contextual_batch,
        )

        receipt = benchmark_contextual_batch(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.reference_receipt,
            args.long_context_receipt,
            args.output,
            layer=args.layer,
            context=args.context,
            cycle_id=args.cycle_id,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "contextual-continuous":
        from swarm_inference.experiments.experiment_014.contextual_continuous import (
            benchmark_contextual_continuous,
        )

        receipt = benchmark_contextual_continuous(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.graph_certification,
            args.reference_receipt,
            args.long_context_receipt,
            args.static_contextual_receipt,
            args.prior_continuous_receipt,
            args.output,
            layer=args.layer,
            context=args.context,
            batch=args.batch,
            cycle_id=args.cycle_id,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "coarse-stage-tcp":
        from swarm_inference.experiments.experiment_014.coarse_stage_transport import (
            benchmark_coarse_stage_tcp,
        )

        receipt = benchmark_coarse_stage_tcp(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.graph_certification,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cycle_id=args.cycle_id,
            ready_timeout_seconds=args.ready_timeout_seconds,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "coarse-network-analysis":
        from swarm_inference.experiments.experiment_014.coarse_network_analysis import (
            analyze_coarse_network,
        )

        receipt = analyze_coarse_network(
            args.coarse_receipt,
            args.fine_receipt,
            args.output,
            args.csv,
            args.chart,
            args.edge_class,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "coarse-boundary-formats":
        from swarm_inference.experiments.experiment_014.boundary_format import (
            benchmark_coarse_boundary_formats,
        )

        receipt = benchmark_coarse_boundary_formats(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.graph_certification,
            args.coarse_receipt,
            args.output,
            device=args.device,
            timing_iterations=args.timing_iterations,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "batched-router":
        from swarm_inference.experiments.experiment_014.router_batch import (
            benchmark_batched_router,
        )

        receipt = benchmark_batched_router(
            args.checkpoint,
            args.candidate_library,
            args.prior_library,
            args.oracle_trace,
            args.output,
            layer=args.layer,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "dense-row-reuse":
        from swarm_inference.experiments.experiment_014.dense_row_reuse import (
            benchmark_dense_row_reuse,
        )

        receipt = benchmark_dense_row_reuse(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.output,
            layer=args.layer,
            role=args.role,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "expert-microwork":
        from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
            benchmark_real_expert_microwork,
        )

        receipt = benchmark_real_expert_microwork(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.graph_certification,
            args.output,
            layer=args.layer,
            workers=args.workers,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cycle_id=args.cycle_id,
            maximum_critical_worker_device_ms=(args.maximum_critical_worker_device_ms),
            minimum_relative_throughput=args.minimum_relative_throughput,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "sub-layer-analysis":
        from swarm_inference.experiments.experiment_014.sub_layer_analysis import (
            analyze_sub_layer_scaling,
        )

        receipt = analyze_sub_layer_scaling(
            {
                2: args.receipt_2,
                4: args.receipt_4,
                8: args.receipt_8,
                16: args.receipt_16,
            },
            args.output,
            args.csv,
            args.chart,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "sub-layer-batch":
        from swarm_inference.experiments.experiment_014.sub_layer_batch import (
            benchmark_sub_layer_batch,
        )

        receipt = benchmark_sub_layer_batch(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.graph_certification,
            args.output,
            layer=args.layer,
            workers=args.workers,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cycle_id=args.cycle_id,
            overlap_parent_shared=args.overlap_parent_shared,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "shared-expert-placement":
        from swarm_inference.experiments.experiment_014.shared_expert_placement import (
            analyze_shared_expert_placement,
        )

        receipt = analyze_shared_expert_placement(
            args.checkpoint,
            args.complete_batch_receipt,
            args.distributed_batch_receipt,
            args.fine_network_receipt,
            args.output,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "sub-layer-recovery":
        from swarm_inference.experiments.experiment_014.sub_layer_recovery import (
            benchmark_sub_layer_recovery,
        )

        receipt = benchmark_sub_layer_recovery(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.passing_batch_receipt,
            args.output,
            device=args.device,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "sub-layer-routing":
        from swarm_inference.experiments.experiment_014.sub_layer_routing import (
            analyze_sub_layer_routing,
        )

        receipt = analyze_sub_layer_routing(
            args.oracle_routes,
            args.serial_receipt,
            args.recovery_receipt,
            args.output,
            args.csv,
            workers=args.workers,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "prefill-stage":
        from swarm_inference.experiments.experiment_014.prefill_stage import (
            benchmark_prefill_stage,
        )

        receipt = benchmark_prefill_stage(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.reference_receipt,
            args.output,
            layer=args.layer,
            contexts=tuple(args.contexts),
            device=args.device,
            cycle_id=args.cycle_id,
            terminal_ratio_floor=args.terminal_ratio_floor,
            terminal_ratio_gate=args.terminal_ratio_gate,
            throughput_retention_gate=args.throughput_retention_gate,
            candidate_regression=args.candidate_regression,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "model-preregister":
        from swarm_inference.experiments.experiment_014.performance_model import (
            preregister_performance_model,
        )

        receipt = preregister_performance_model(
            args.depth_profile,
            args.mla_context_profile,
            args.cuda_library,
            args.output,
            cycle_id=args.cycle_id,
            median_ape_gate_percent=args.median_ape_gate_percent,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "model-validate":
        from swarm_inference.experiments.experiment_014.performance_model import (
            validate_performance_model,
        )

        receipt = validate_performance_model(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.reference_receipt,
            args.preregistration,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "capacity-topology":
        from swarm_inference.experiments.experiment_014.capacity_topology import (
            build_capacity_topology_model,
        )

        receipt = build_capacity_topology_model(
            {
                "validation": args.validation,
                "resident_profile": args.resident_profile,
                "depth_profile": args.depth_profile,
                "stage_zero": args.stage_zero,
                "final_stage": args.final_stage,
                "lm_head": args.lm_head,
                "complete_kda_batch": args.complete_kda_batch,
                "complete_mla_batch": args.complete_mla_batch,
                "contextual_static": args.contextual_static,
                "contextual_route_mix": args.contextual_route_mix,
                "contextual_matched": args.contextual_matched,
                "kda_prefill": args.kda_prefill,
                "mla_prefill": args.mla_prefill,
                "sub_layer_scaling": args.sub_layer_scaling,
                "sub_layer_batch": args.sub_layer_batch,
                "coarse_transport": args.coarse_transport,
                "coarse_network": args.coarse_network,
                "node_solver": args.node_solver,
            },
            args.output,
            args.candidate_csv,
            args.economics_csv,
            args.chart,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "final-placement":
        from swarm_inference.experiments.experiment_014.final_deployment import (
            build_final_placement,
        )

        receipt = build_final_placement(
            args.old_placement,
            args.capacity_receipt,
            args.stage_zero_receipt,
            args.kda_batch_receipt,
            args.mla_batch_receipt,
            args.final_stage_receipt,
            args.cuda_library,
            args.promotion_receipt,
            args.output,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "acquisition-fixture":
        from swarm_inference.experiments.experiment_014.remote_acquisition import (
            benchmark_remote_acquisition_fixture,
        )

        receipt = benchmark_remote_acquisition_fixture(
            args.output, cycle_id=args.cycle_id
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "tokenizer-product-fixture":
        from swarm_inference.experiments.experiment_014.remote_acquisition import (
            benchmark_kimi_tokenizer_product_seam,
        )

        receipt = benchmark_kimi_tokenizer_product_seam(
            args.tokenizer_directory,
            args.conversation_receipt,
            args.output,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "acquire-worker":
        from swarm_inference.experiments.experiment_014.remote_acquisition import (
            acquire_worker_package,
        )

        receipt = acquire_worker_package(
            args.distribution_manifest,
            args.worker_id,
            args.cache_directory,
            args.output,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "activate-worker-snapshot":
        from swarm_inference.experiments.experiment_014.remote_acquisition import (
            activate_worker_snapshot,
        )

        receipt = activate_worker_snapshot(
            args.placement,
            args.worker_id,
            args.package,
            args.config,
            args.output_directory,
            model_id=args.model_id,
            tokenizer_revision=args.tokenizer_revision,
            adapter_id=args.adapter_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "deployment-recovery":
        from swarm_inference.experiments.experiment_014.deployment_recovery import (
            benchmark_coarse_stage_recovery,
        )

        receipt = benchmark_coarse_stage_recovery(
            args.output, cycle_id=args.cycle_id
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "admission-fixture":
        from swarm_inference.experiments.experiment_014.deployment_admission import (
            run_admission_fixture,
        )

        receipt = run_admission_fixture(args.output, cycle_id=args.cycle_id)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "node-admission":
        from swarm_inference.experiments.experiment_014.deployment_admission import (
            run_node_admission,
        )

        receipt = run_node_admission(
            args.observation,
            args.requirements,
            args.output,
            mode=args.mode,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "fleet-cost-guard":
        from swarm_inference.experiments.experiment_014.deployment_admission import (
            run_fleet_cost_guard,
        )

        receipt = run_fleet_cost_guard(
            args.fleet_status, args.policy, args.output
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "ALLOW" else 78
    if args.command == "worker-requirements":
        from swarm_inference.experiments.experiment_014.deployment_admission import (
            write_worker_requirements,
        )

        receipt = write_worker_requirements(
            args.placement,
            args.distribution,
            args.worker_id,
            args.output,
            package_version=args.package_version,
            runtime_sha256=args.runtime_sha256,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "inspect-deployment-node":
        from swarm_inference.experiments.experiment_014.deployment_admission import (
            write_local_node_observation,
        )

        receipt = write_local_node_observation(
            args.output,
            runtime_path=args.runtime,
            network_rtt_ms=args.network_rtt_ms,
            network_bandwidth_gbps=args.network_bandwidth_gbps,
            assignment_sha256=args.assignment_sha256,
            checkpoint_revision=args.checkpoint_revision,
            package_version=args.package_version,
            disk_path=args.disk_path,
            device=args.device,
            assigned_stage_canary_path=args.assigned_stage_canary,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "bind-deployment-plan":
        from swarm_inference.experiments.experiment_014.deployment_package import (
            build_bound_product_plan,
        )

        receipt = build_bound_product_plan(
            args.placement,
            args.workers_status,
            args.output,
            runtime_certificate_path=args.runtime_certificate,
            native_source_manifest_path=args.native_source_manifest,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "deployment-identity-fixture":
        from swarm_inference.experiments.experiment_014.deployment_package import (
            benchmark_deployment_identity_fixture,
        )

        receipt = benchmark_deployment_identity_fixture(
            args.placement,
            args.output,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "build-physical-canary-fixtures":
        from swarm_inference.experiments.experiment_014.deployment_canary import (
            build_physical_canary_fixtures,
        )

        receipt = build_physical_canary_fixtures(
            args.checkpoint,
            args.oracle_trace,
            args.oracle_routes,
            args.oracle_logits,
            args.placement,
            args.output_npz,
            args.output_manifest,
            full_graph_receipt=args.full_graph_receipt,
            operation_matrix_receipt=args.operation_matrix_receipt,
            sm86_receipt=args.sm86_receipt,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "physical-3090-canary":
        from swarm_inference.experiments.experiment_014.deployment_canary import (
            run_physical_canary,
        )

        receipt = run_physical_canary(
            args.placement,
            args.native_source_manifest,
            args.runtime,
            args.fixture_npz,
            args.fixture_manifest,
            args.evidence_root,
            {
                "stage_zero_embedding_dense": args.stage_zero_snapshot,
                "kda_moe": args.kda_snapshot,
                "gated_mla_moe": args.mla_snapshot,
                "final_norm_head_sampling": args.final_snapshot,
            },
            args.receipt,
            args.certificate,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "assigned-stage-canary":
        from swarm_inference.experiments.experiment_014.deployment_canary import (
            run_assigned_stage_canary,
        )

        receipt = run_assigned_stage_canary(
            args.placement,
            args.native_source_manifest,
            args.runtime_certificate,
            args.runtime,
            args.snapshot,
            args.worker_id,
            args.output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "build-experiment-015-package":
        from swarm_inference.experiments.experiment_014.experiment_015_package import (
            build_experiment_015_package,
        )

        receipt = build_experiment_015_package(
            args.repository_root,
            args.output,
            args.wheel,
            args.checkpoint_metadata_directory,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "validate-experiment-015-package":
        from swarm_inference.experiments.experiment_014.experiment_015_package import (
            validate_experiment_015_package,
        )

        receipt = validate_experiment_015_package(args.package)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "experiment-015-package-controls":
        from swarm_inference.experiments.experiment_014.experiment_015_package import (
            benchmark_package_negative_controls,
        )

        receipt = benchmark_package_negative_controls(
            args.package, args.output, cycle_id=args.cycle_id
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 78
    if args.command == "batch-scaling":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_batch_scaling,
        )

        receipt = benchmark_batch_scaling(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.batch2_profile,
            args.output,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "expert-working-set":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_expert_working_set,
        )

        receipt = benchmark_expert_working_set(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.steady_profile,
            args.output,
            device=args.device,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "device-warmup":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_device_warmup,
        )

        receipt = benchmark_device_warmup(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.steady_profile,
            args.cold_profile,
            args.output,
            device=args.device,
            minimum_warmup_device_ms=args.minimum_warmup_device_ms,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "integrated-readiness":
        from swarm_inference.experiments.experiment_014.performance import (
            benchmark_integrated_readiness,
        )

        receipt = benchmark_integrated_readiness(
            args.checkpoint,
            args.cuda_library,
            args.oracle_trace,
            args.oracle_routes,
            args.identity_manifest,
            args.steady_profile,
            args.output,
            device=args.device,
            cycle_id=args.cycle_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "distribution":
        receipt = build_distribution_manifest(args.checkpoint, args.placement, args.output)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1
    if args.command == "materialize-worker":
        distribution_manifest = json.loads(args.distribution_manifest.read_text(encoding="utf-8"))
        source_hashes = {
            name: row["sha256"] for name, row in distribution_manifest["source"]["shards"].items()
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
