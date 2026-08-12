"""Generate the complete, integrity-linked Experiment 015 handoff."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.architecture_search import (
    search_architectures,
)
from swarm_inference.experiments.experiment_015.baseline import capture_baseline
from swarm_inference.experiments.experiment_015.contracts import (
    EconomicsConfig,
    EvidenceClass,
)
from swarm_inference.experiments.experiment_015.dcp import benchmark_dcp_combination
from swarm_inference.experiments.experiment_015.evidence import (
    atomic_json,
    atomic_text,
    file_identity,
    read_json,
)
from swarm_inference.experiments.experiment_015.figures import generate_figures
from swarm_inference.experiments.experiment_015.microwork import analyze_microwork
from swarm_inference.experiments.experiment_015.network import (
    NetworkProfile,
    activation_payload_bytes,
)
from swarm_inference.experiments.experiment_015.packing import build_packing_model
from swarm_inference.experiments.experiment_015.pipeline import (
    analyze_pipeline_occupancy,
)
from swarm_inference.experiments.experiment_015.routing import (
    analyze_routes,
    export_route_rows,
)
from swarm_inference.experiments.experiment_015.service_model import (
    ArchitectureServiceModel,
    load_service_evidence,
    validate_service_model,
)
from swarm_inference.experiments.experiment_015.tp import analyze_targeted_tp


def _source_ledger(root: Path, destination: Path) -> dict[str, Any]:
    vllm_root = root / "artifacts" / "experiment-015" / "research-cache" / "vllm-source"
    dspark_metadata = (
        root / "artifacts" / "experiment-015" / "research-cache" / "dspark-metadata"
    )
    entries = [
        {
            "id": "R015-001",
            "topic": "Kimi K3 serving and DSpark/DCP",
            "title": "Kimi K3 Is Here: Efficient Day-0 Support on vLLM",
            "url": "https://vllm-project.github.io/2026/07/27/k3.html",
            "source_type": "official project engineering report",
            "used_for": "hypothesis motivation and external reference only",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-002",
            "topic": "Kimi K3 DSpark",
            "title": "Inferact/Kimi-K3-DSpark model card",
            "url": "https://huggingface.co/Inferact/Kimi-K3-DSpark/blob/main/README.md",
            "source_type": "public model card",
            "used_for": "checkpoint identity, architecture, public acceptance sensitivity",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-003",
            "topic": "DSpark",
            "title": "DSpark: Confidence-Scheduled Speculative Decoding",
            "url": "https://arxiv.org/abs/2607.05147",
            "source_type": "research paper",
            "used_for": "confidence scheduling and verification-waste hypothesis",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-004",
            "topic": "speculative decoding",
            "title": "Fast Inference from Transformers via Speculative Decoding",
            "url": "https://proceedings.mlr.press/v202/leviathan23a.html",
            "source_type": "peer-reviewed paper",
            "used_for": "distribution-preserving rejection-sampling semantics",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-005",
            "topic": "tree speculative inference",
            "title": "SpecInfer",
            "url": "https://arxiv.org/abs/2305.09781",
            "source_type": "research paper",
            "used_for": "token-tree branch criterion",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-006",
            "topic": "pipeline speculative inference",
            "title": "PipeInfer",
            "url": "https://arxiv.org/abs/2407.11798",
            "source_type": "research paper",
            "used_for": "asynchronous speculation and early cancellation hypothesis",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-007",
            "topic": "hierarchical speculative pipelines",
            "title": "PipeSpec",
            "url": "https://aclanthology.org/2025.findings-acl.669/",
            "source_type": "peer-reviewed paper",
            "used_for": "hierarchical pipeline hypothesis",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-008",
            "topic": "expert parallelism",
            "title": "DeepEP",
            "url": "https://github.com/deepseek-ai/DeepEP",
            "source_type": "official open-source implementation",
            "used_for": "device-resident dispatch/combine and overlap principles",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-009",
            "topic": "expert load balancing",
            "title": "Expert Parallelism Load Balancer",
            "url": "https://github.com/deepseek-ai/EPLB",
            "source_type": "official open-source implementation",
            "used_for": "redundant expert placement hypothesis",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-010",
            "topic": "Decode Context Parallelism",
            "title": "vLLM Context Parallel Deployment",
            "url": "https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/",
            "source_type": "official documentation",
            "used_for": "DCP exact-combine and conditional-admission design",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-011",
            "topic": "on-demand expert loading",
            "title": "OD-MoE",
            "url": "https://arxiv.org/abs/2512.03927",
            "source_type": "research paper",
            "used_for": "tiered residency hypothesis",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-012",
            "topic": "hybrid expert offload",
            "title": "HybriMoE",
            "url": "https://arxiv.org/abs/2504.05897",
            "source_type": "research paper",
            "used_for": "dynamic CPU/GPU scheduling, prefetch, and caching hypothesis",
            "imported_into_swarm_result": False,
        },
        {
            "id": "R015-013",
            "topic": "communication/computation overlap",
            "title": "DeepSeek public profiling data",
            "url": "https://github.com/deepseek-ai/profile-data",
            "source_type": "official profiling repository",
            "used_for": "operation-timeline/nanobatching hypothesis",
            "imported_into_swarm_result": False,
        },
    ]
    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-research-source-ledger-v1",
        "status": "PASS",
        "accessed_date": "2026-08-12",
        "policy": "external work motivates hypotheses; only Swarm evidence determines conclusions",
        "entries": entries,
        "pinned_local_research_material": {
            "vllm_commit": "a53ad859139ee37ef7c2c29716963abffd9cb486",
            "vllm_dspark_pipeline_guard": file_identity(
                vllm_root / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "dspark" / "utils.py",
                relative_to=root,
            ),
            "dspark_model_card": file_identity(dspark_metadata / "README.md", relative_to=root),
            "dspark_config": file_identity(dspark_metadata / "config.json", relative_to=root),
        },
    }
    atomic_json(destination, receipt)
    lines = [
        "# Experiment 015 research source ledger",
        "",
        "External results motivate hypotheses only; none are relabelled as Swarm measurements.",
        "",
        "| ID | Topic | Source | Use | Imported into Swarm result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['id']} | {entry['topic']} | [{entry['title']}]({entry['url']}) | "
            f"{entry['used_for']} | NO |"
        )
    atomic_text(destination.with_suffix(".md"), "\n".join(lines))
    return receipt


def _speculation_artifacts(root: Path, artifact_root: Path) -> dict[str, Any]:
    sweep_path = artifact_root / "dspark" / "h015-001b-dspark-reference-sweep.json"
    repeat_path = artifact_root / "dspark" / "h015-001c-dspark-reference-repeat.json"
    sweep = read_json(sweep_path)
    workloads = [
        {
            "workload_id": "coding-001",
            "class": "coding",
            "prompt": "Implement a deterministic LRU cache in Python and explain its invariants.",
        },
        {
            "workload_id": "math-001",
            "class": "mathematical/reasoning",
            "prompt": "Prove that the sum of the first n odd integers is n squared.",
        },
        {
            "workload_id": "chat-001",
            "class": "general chat",
            "prompt": "Explain why leaves change colour in autumn to a curious teenager.",
        },
        {
            "workload_id": "creative-001",
            "class": "creative/high-entropy",
            "prompt": "Write an original scene set in a lighthouse during an impossible tide.",
        },
    ]
    manifest = {
        "schema_version": "experiment-015-workload-manifest-v1",
        "status": "PASS",
        "deterministic": True,
        "seed": 15015,
        "sampling": {"mode": "greedy", "temperature": 0.0},
        "maximum_output_tokens": 64,
        "workloads": workloads,
    }
    atomic_json(artifact_root / "speculation" / "workload-manifest.json", manifest)

    traces: list[dict[str, Any]] = []
    for workload in workloads:
        for block_size in (1, 2, 3, 5, 7):
            traces.append(
                {
                    "workload_id": workload["workload_id"],
                    "workload_class": workload["class"],
                    "block_size": block_size,
                    "status": "NOT_EXECUTED",
                    "accepted_tokens": None,
                    "target_passes": None,
                    "target_traversals_per_output_token": None,
                    "reason": "representative real-target verification was not completed",
                }
            )
    acceptance = {
        "schema_version": "experiment-015-speculation-acceptance-traces-v1",
        "cycle_id": "H015-001B",
        "status": "INCOMPLETE",
        "evidence_class": None,
        "scientific_result": False,
        "measurement_count": 0,
        "rows": traces,
        "public_reference_not_swarm": {
            "block_size": 7,
            "greedy_mean_accepted_length": 3.85,
            "stochastic_mean_accepted_length": 3.73,
        },
    }
    acceptance_path = artifact_root / "speculation" / "acceptance-traces.json"
    atomic_json(acceptance_path, acceptance)
    csv_path = acceptance_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)

    dspark_result = {
        "schema_version": "experiment-015-dspark-results-v1",
        "cycle_id": "H015-001A-H015-001B",
        "status": "FAIL",
        "reference_execution": {
            "evidence_class": sweep["evidence_class"],
            "scientific_result": sweep["scientific_result"],
            "scope": sweep["measurement_scope"],
            "blocks": sweep["blocks"],
            "correctness": sweep["correctness"],
        },
        "checkpoint": sweep["draft"],
        "acceptance": {
            "swarm_mean_accepted_length": None,
            "target_traversals_per_output_token": None,
            "first_gate_2x": "NOT_EVALUABLE",
        },
        "decision": "STOP_STANDARD_DSPARK_GATE_UNRESOLVED",
        "limiting_mechanism": (
            "one streamed 93-layer target traversal takes roughly an hour locally, "
            "and the pinned public runtime rejects DSpark pipeline parallelism"
        ),
        "sources": {
            "reference_sweep": file_identity(sweep_path, relative_to=root),
            "repeat": file_identity(repeat_path, relative_to=root),
            "acceptance_trace": file_identity(acceptance_path, relative_to=root),
        },
    }
    atomic_json(artifact_root / "dspark" / "results.json", dspark_result)
    return dspark_result


def _dcp_artifacts(root: Path, artifact_root: Path) -> dict[str, Any]:
    correctness = benchmark_dcp_combination(artifact_root / "dcp" / "correctness.json")
    evidence = load_service_evidence(root)
    model = ArchitectureServiceModel(evidence)
    performance_rows = [
        model.dcp_stage_projection(context_tokens=context, degree=degree)
        for context in (1024, 4096, 8192, 16384)
        for degree in (1, 2, 4, 8)
    ]
    row_8k_8 = next(
        row
        for row in performance_rows
        if row["context_tokens"] == 8192 and row["degree"] == 8
    )
    receipt = {
        "schema_version": "experiment-015-dcp-results-v1",
        "cycle_id": "H015-006A-H015-006B",
        "status": "FAIL",
        "component_correctness": correctness,
        "performance_rows": performance_rows,
        "best_8k_sensitivity": row_8k_8,
        "complete_kimi_correctness": "NOT_ESTABLISHED",
        "complete_kimi_performance": "NOT_ESTABLISHED",
        "beneficial_context_threshold": "NOT_ESTABLISHED",
        "best_degree": "NOT_ESTABLISHED",
        "decision": "STOP_NO_REAL_DISTRIBUTED_KIMI_DCP_KERNEL",
    }
    atomic_json(artifact_root / "dcp" / "results.json", receipt)
    return receipt


def _network_and_capacity(root: Path, artifact_root: Path) -> None:
    payload = activation_payload_bytes(1)
    bandwidths = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0)
    domains: dict[str, Any] = {}
    for name, rtts in (
        ("internal_microwork", (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)),
        ("coarse_inter_cell", (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)),
    ):
        rows = []
        for rtt in rtts:
            for bandwidth in bandwidths:
                profile = NetworkProfile(name, rtt_ms=rtt, bandwidth_gbps=bandwidth)
                rows.append(
                    {
                        "evidence_class": "SHAPED",
                        "rtt_ms": rtt,
                        "bandwidth_gbps": bandwidth,
                        "payload_bytes": payload,
                        "service_ms": profile.service_ms(payload),
                    }
                )
        domains[name] = rows
    atomic_json(
        artifact_root / "network-surfaces" / "activation-boundaries.json",
        {
            "schema_version": "experiment-015-network-surfaces-v1",
            "status": "PASS",
            "actual_kimi_fp32_boundary_payload_bytes": payload,
            "software_base_ms_from_retained_loopback": 0.6102,
            "domains": domains,
            "admission_note": "internal and coarse domains remain separate planner inputs",
        },
    )

    architecture = read_json(artifact_root / "architecture-pareto" / "results.json")
    baseline = read_json(artifact_root / "baseline" / "B015-000.json")
    spec_rows = []
    model = ArchitectureServiceModel(load_service_evidence(root))
    for depth in (1, 2, 4, 8):
        for block in (1, 2, 3, 5, 7):
            spec_rows.append(
                model.speculation_upper_bound(
                    block_size=block,
                    accepted_tokens_per_target_pass=float(block + 1),
                    cell_depth=depth,
                )
            )
    atomic_json(
        artifact_root / "capacity-model" / "results.json",
        {
            "schema_version": "experiment-015-capacity-model-v1",
            "status": "PASS",
            "evidence_class": "SHAPED",
            "service_model_validation": (
                "component model passed held-out validation; topology transport is synthetic"
            ),
            "baseline": baseline["performance"],
            "microcells": architecture["microcell_sweep"],
            "perfect_acceptance_upper_bounds": spec_rows,
            "critical_upper_bound": architecture["perfect_acceptance_upper_bound"],
            "conclusion": "even block-7 perfect acceptance plus depth-8 cells remains below 5 tok/s",
        },
    )


def _cycles(artifact_root: Path) -> list[dict[str, Any]]:
    architecture = read_json(artifact_root / "architecture-pareto" / "results.json")
    micro = read_json(artifact_root / "microwork" / "results.json")
    dcp = read_json(artifact_root / "dcp" / "results.json")
    routing = read_json(artifact_root / "expert-routing" / "results.json")
    predictor = routing["previous_token_same_layer_predictor"]
    return [
        {
            "id": "H015-001A",
            "hypothesis": "Pinned public Kimi K3 DSpark can draft deterministic candidate blocks from real target hidden states.",
            "expected_quantitative_result": "finite deterministic blocks 1/2/3/5/7 and exact repeat",
            "implementation": "CPU BF16 public-weight reference plus transactional commit/rollback semantics",
            "benchmark": "pinned revisions/hashes; retained real Kimi hidden trace; blocks 1/2/3/5/7",
            "result": "finite deterministic blocks and repeat passed; acceptance/distribution equivalence not established",
            "inspection": "real DSpark weights executed; target verification was absent",
            "bottleneck": "representative 93-layer target pass cost and no public DSpark pipeline support",
            "decision": "MODIFY",
            "redesign": "bound target verification before implementing pipeline speculation",
        },
        {
            "id": "H015-001B",
            "hypothesis": "Block-7 verification can reach at least 5 tok/s with perfect acceptance after boundary reduction.",
            "expected_quantitative_result": ">=5 dependency-bound tok/s",
            "implementation": "shaped target-batch service upper bound with depth-8 cells and zero draft cost",
            "benchmark": "immutable batch-8 KDA/MLA timings; perfect 8 outputs/pass",
            "result": f"{architecture['perfect_acceptance_upper_bound']['dependency_bound_tok_s']:.4f} tok/s",
            "inspection": "target verification widens each pass to batch 8 and raises traversal service time",
            "bottleneck": "target block-verification service, not only boundary latency",
            "decision": "REVERT",
            "redesign": "a faster verification kernel/architecture is required before DSpark can hit 5 tok/s",
        },
        {
            "id": "H015-002A",
            "hypothesis": "Asynchronous drafting improves ordinary DSpark after the synchronous path passes.",
            "expected_quantitative_result": ">10% over synchronous DSpark at equal target work",
            "implementation": "occupancy/admission contract; no distributed runtime mutation",
            "benchmark": "baseline stage-time model and pinned vLLM DSpark pipeline guard",
            "result": "NOT ESTABLISHED",
            "inspection": "ordinary DSpark acceptance/GPU draft latency gate failed first",
            "bottleneck": "missing synchronous baseline and unsupported public PP combination",
            "decision": "REVERT",
            "redesign": "resume only after H015-001 passes",
        },
        {
            "id": "H015-003A",
            "hypothesis": "Depth-4/8 cells materially reduce exposed slow-boundary latency.",
            "expected_quantitative_result": ">=20% dependency speed improvement for depth 8",
            "implementation": "separate internal/coarse domains in a held-out-validated component model with shaped network",
            "benchmark": "depth 1/2/4/8, actual 258,048-byte Kimi boundary",
            "result": f"depth 8: {architecture['best_justified_result']['dependency_bound_tok_s']:.4f} tok/s, 23.8% faster",
            "inspection": "compute and 81 internal boundaries remain sequential",
            "bottleneck": "747 ms compute plus internal synchronization",
            "decision": "RETAIN",
            "redesign": "combine only with a verification method that reduces target passes",
        },
        {
            "id": "H015-004A",
            "hypothesis": "Direct resident expert dispatch achieves >=90% complete-layer throughput.",
            "expected_quantitative_result": ">=90% measured relative throughput",
            "implementation": "overhead decomposition and real hidden-row transport precision checks",
            "benchmark": "immutable 2/4/8/16 worker real Kimi sweep; promoted four-worker repeat",
            "result": f"78.81% measured; {100*micro['direct_resident_redesign_upper_bound']['optimistic_relative_throughput']:.2f}% optimistic subtraction bound",
            "inspection": "host/network/serialization costs are material but replacement cost was not measured",
            "bottleneck": "missing device-resident transport implementation",
            "decision": "REVERT",
            "redesign": "implement persistent device buffers before another performance claim",
        },
        {
            "id": "H015-005A",
            "hypothesis": "Expert transport overlap hides a significant fraction of communication.",
            "expected_quantitative_result": ">=25% exposed communication hidden",
            "implementation": "admission surface only",
            "benchmark": "retained timing decomposition",
            "result": "NOT ESTABLISHED",
            "inspection": "no asynchronous send/receive implementation ran",
            "bottleneck": "lack of independent resident network/compute streams",
            "decision": "REVERT",
            "redesign": "requires H015-004 resident buffers first",
        },
        {
            "id": "H015-006A",
            "hypothesis": "Exact context-shard sufficient statistics reproduce full softmax attention.",
            "expected_quantitative_result": "relative L2 <=1e-12 for degrees 1/2/4/8",
            "implementation": "stable max/denominator/numerator reduction",
            "benchmark": "deterministic 1K synthetic context, uneven-capable shards",
            "result": f"maximum relative L2 {dcp['component_correctness']['maximum_relative_l2_error']:.3e}",
            "inspection": "component exactness passed; complete Kimi state path not exercised",
            "bottleneck": "no real distributed Kimi MLA kernel",
            "decision": "MODIFY",
            "redesign": "integrate into persistent MLA worker before capacity claims",
        },
        {
            "id": "H015-006B",
            "hypothesis": "DCP4+ reduces 8K MLA service by >=25%.",
            "expected_quantitative_result": ">=25% shaped stage gain",
            "implementation": "measured context-scan fit plus actual partial payload network model",
            "benchmark": "1K/4K/8K/16K x DCP1/2/4/8",
            "result": f"8K DCP8 shaped gain {100*dcp['best_8k_sensitivity']['stage_gain_fraction']:.2f}%",
            "inspection": "worker cost/capacity and complete-stage correctness remain unknown",
            "bottleneck": "distributed combine and paid-compute validation",
            "decision": "REVERT",
            "redesign": "real Kimi DCP kernel and conditional context gate",
        },
        {
            "id": "H015-007A",
            "hypothesis": "Only operations larger than their collectives benefit from TP.",
            "expected_quantitative_result": ">1.0x complete-operation speedup at TP2/4",
            "implementation": "real CUDA TP1 timings plus shaped internal collective",
            "benchmark": "KDA q/output, MLA projection group, LM head; TP1/2/4",
            "result": "LM head TP4 sensitivity 1.63x; KDA/MLA candidates did not win",
            "inspection": "isolated head gain is small in the 1,052 ms model path",
            "bottleneck": "collective overhead and lack of full-stage implementation",
            "decision": "REVERT",
            "redesign": "retain LM-head TP as a later endpoint-only candidate",
        },
        {
            "id": "H015-008A",
            "hypothesis": "Measured load placement reduces critical expert worker demand.",
            "expected_quantitative_result": ">=10% held-out critical-load reduction",
            "implementation": "static modulo and greedy trace-fitted placement",
            "benchmark": "three real tokens across 92 MoE layers",
            "result": f"{100*routing['trace_fitted_balanced_placement']['critical_load_gain_fraction']:.2f}% in-sample gain; no held-out trace",
            "inspection": "fit and evaluation reused the same tiny trace",
            "bottleneck": "representative route history",
            "decision": "REVERT",
            "redesign": "collect multi-workload routes before replication/migration",
        },
        {
            "id": "H015-009A",
            "hypothesis": "Previous-token routes predict enough experts to hide dispatch.",
            "expected_quantitative_result": ">=70% precision and recall",
            "implementation": "same-layer previous-token top-16 predictor",
            "benchmark": "184 transitions in retained route trace",
            "result": f"precision/recall {100*predictor['precision']:.2f}%",
            "inspection": f"{100*(1-predictor['precision']):.2f}% of predicted activation bytes were wasted",
            "bottleneck": "low temporal route overlap",
            "decision": "REVERT",
            "redesign": "do not train a learned predictor until representative traces exist",
        },
        {
            "id": "H015-010A",
            "hypothesis": "25-75% expert residency improves paid-GPU efficiency without decode collapse.",
            "expected_quantitative_result": "positive tok/s/$ after H2D misses",
            "implementation": "cold-start per-layer LRU replay and transfer accounting",
            "benchmark": "100/75/50/25% residency over three target tokens",
            "result": "NOT ESTABLISHED",
            "inspection": "three tokens cannot estimate steady-state hit rate or throughput",
            "bottleneck": "representative route locality and real H2D overlap",
            "decision": "REVERT",
            "redesign": "collect long traces before whole-expert or partial-expert residency",
        },
        {
            "id": "H015-011A",
            "hypothesis": "DSpark plus microcells exceeds both primary targets.",
            "expected_quantitative_result": ">=5 tok/s and >=2.78 tok/s/GPU-equivalent",
            "implementation": "admission-gated architecture model; no unpassed components multiplied",
            "benchmark": "A-M search plus perfect-acceptance upper bound",
            "result": "perfect block-7/depth-8 bound 2.18 tok/s and 1.04 tok/s/GPU excluding draft",
            "inspection": "verification batch service consumes the same rows used by aggregate batch 8",
            "bottleneck": "target verification service and unchanged paid compute",
            "decision": "REVERT",
            "redesign": "requires fundamentally faster target verification, not another topology-only combination",
        },
        {
            "id": "H015-012A",
            "hypothesis": "Operation nanobatching improves utilization without violating 5 tok/s cadence.",
            "expected_quantitative_result": ">10% throughput at dependency >=5 tok/s",
            "implementation": "prerequisite gate only",
            "benchmark": "immutable safe batch 8 capacity and dependency model",
            "result": "NOT EXECUTED; dependency prerequisite failed",
            "inspection": "baseline already exploits batch 8 for aggregate capacity",
            "bottleneck": "individual dependency path",
            "decision": "REVERT",
            "redesign": "resume only after a >=5 tok/s architecture exists",
        },
        {
            "id": "H015-PRIMARY",
            "hypothesis": "A speculative hierarchical sub-layer architecture materially improves both axes.",
            "expected_quantitative_result": ">=5 tok/s/user and >=2.78 tok/s/GPU-equivalent",
            "implementation": "immutable baseline, component gates, held-out validation, shaped topology, admission-gated A-M search",
            "benchmark": "all retained local evidence and explicit upper bounds",
            "result": "FAIL: best admitted projection 1.1766 tok/s and 1.0446 tok/s/GPU-equivalent",
            "inspection": "no speculative/EP/DCP/TP/residency component passed complete implementation gates",
            "bottleneck": "multi-token target verification service plus unresolved distributed implementations",
            "decision": "REVERT",
            "redesign": "do not proceed to physical validation; build a local complete verifier first",
        },
    ]


def _cycle_ledger(artifact_root: Path) -> None:
    cycles = _cycles(artifact_root)
    atomic_json(
        artifact_root / "cycle-ledger.json",
        {
            "schema_version": "experiment-015-cycle-ledger-v1",
            "status": "PASS",
            "cycles": cycles,
        },
    )
    lines = [
        "# Experiment 015 cycle ledger",
        "",
        "Failed and stopped branches are retained. Results are not promoted across evidence classes.",
        "",
        "| Field | Required evidence |",
        "| --- | --- |",
    ]
    for cycle in cycles:
        lines.extend(
            [
                f"| ID | {cycle['id']} |",
                f"| Hypothesis | {cycle['hypothesis']} Expected: {cycle['expected_quantitative_result']} |",
                f"| Implementation | {cycle['implementation']} |",
                f"| Benchmark | {cycle['benchmark']} |",
                f"| Result | {cycle['result']} |",
                f"| Inspection | {cycle['inspection']} |",
                f"| Bottleneck | {cycle['bottleneck']} |",
                f"| Decision | {cycle['decision']} |",
                f"| Redesign | {cycle['redesign']} |",
                "",
            ]
        )
    atomic_text(artifact_root / "cycle-ledger.md", "\n".join(lines))


def _classification_audit(artifact_root: Path) -> dict[str, Any]:
    """Reject invented classes and require null classes to be diagnostic-only."""
    allowed = {item.value for item in EvidenceClass}
    counts = {item.value: 0 for item in EvidenceClass}
    counts["UNCLASSIFIED_DIAGNOSTIC"] = 0
    violations: list[str] = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, dict):
            if "evidence_class" in value:
                evidence_class = value["evidence_class"]
                if evidence_class is None:
                    counts["UNCLASSIFIED_DIAGNOSTIC"] += 1
                    if value.get("scientific_result") is not False:
                        violations.append(
                            f"{location}: null evidence_class without scientific_result=false"
                        )
                elif evidence_class in allowed:
                    counts[str(evidence_class)] += 1
                else:
                    violations.append(
                        f"{location}: invalid evidence_class={evidence_class!r}"
                    )
            for key, child in value.items():
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    for path in artifact_root.rglob("*.json"):
        relative = path.relative_to(artifact_root)
        if relative.parts[0] == "research-cache" or path.name == "evidence-integrity.json":
            continue
        visit(read_json(path), relative.as_posix())
    return {
        "status": "PASS" if not violations else "FAIL",
        "allowed_classes": sorted(allowed),
        "counts": counts,
        "violations": violations,
        "null_class_rule": "scientific_result must be false",
    }


def _gates_and_summary(artifact_root: Path, economics: EconomicsConfig) -> dict[str, Any]:
    architecture = read_json(artifact_root / "architecture-pareto" / "results.json")
    baseline = read_json(artifact_root / "baseline" / "B015-000.json")
    micro = read_json(artifact_root / "microwork" / "results.json")
    dcp = read_json(artifact_root / "dcp" / "results.json")
    validation = read_json(artifact_root / "model-validation" / "results.json")
    full_graph_path = artifact_root / "regression" / "full-93-layer.json"
    full_graph = read_json(full_graph_path) if full_graph_path.is_file() else None
    full_graph_pass = bool(
        full_graph is not None
        and full_graph.get("status") == "PASS"
        and full_graph.get("coverage", {}).get("layers_executed") == 93
        and full_graph.get("correctness", {}).get("routing_equality") is True
        and full_graph.get("correctness", {}).get("stateful_decode_executed") is True
    )
    classification_audit = _classification_audit(artifact_root)
    best = architecture["best_justified_result"]
    gates = [
        {"gate": "B015-000 immutable capture", "status": "PASS"},
        {
            "gate": "evidence classification",
            "status": classification_audit["status"],
            "violations": classification_audit["violations"],
        },
        {
            "gate": "full 93-layer Kimi CUDA regression",
            "status": "PASS" if full_graph_pass else "FAIL",
            "value": (
                full_graph["correctness"]["maximum_layer_relative_l2_error"]
                if full_graph is not None
                else None
            ),
        },
        {"gate": "service model median error <=10%", "status": validation["status"], "value_percent": validation["median_absolute_percentage_error_percent"]},
        {"gate": "DSpark target distribution and representative acceptance", "status": "FAIL"},
        {"gate": "speculation >=2x dependency speed", "status": "FAIL"},
        {"gate": "microcell boundary reduction model", "status": "PASS"},
        {"gate": "expert microwork >=90% measured", "status": "FAIL", "value_percent": 100 * micro["best_measured"]["relative_complete_layer_throughput"]},
        {"gate": "complete Kimi DCP", "status": "FAIL"},
        {"gate": "complete-stage targeted TP", "status": "FAIL"},
        {"gate": "held-out expert placement/replication", "status": "FAIL"},
        {"gate": "route predictor", "status": "FAIL"},
        {"gate": "tiered residency throughput", "status": "FAIL"},
        {"gate": ">=5 dependency tok/s", "status": "FAIL", "value": best["dependency_bound_tok_s"]},
        {"gate": ">=2.78 aggregate tok/s/GPU-equivalent", "status": "FAIL", "value": best["aggregate_tok_s_per_paid_gpu_equivalent"]},
    ]
    atomic_json(
        artifact_root / "acceptance-gates.json",
        {
            "schema_version": "experiment-015-acceptance-gates-v1",
            "status": "FAIL",
            "gates": gates,
        },
    )
    summary = {
        "schema_version": "experiment-015-machine-summary-v1",
        "experiment": "015",
        "status": "FAIL",
        "verdict": "FAIL: unresolved implementation/correctness evidence",
        "best_complete_measured_architecture": "B015-000 control only",
        "best_justified_result": best,
        "metrics": {
            "baseline_dependency_bound_tok_s": baseline["performance"]["dependency_bound_tok_s"],
            "best_dependency_bound_tok_s": best["dependency_bound_tok_s"],
            "dependency_speedup": best["dependency_bound_tok_s"] / baseline["performance"]["dependency_bound_tok_s"],
            "baseline_aggregate_tok_s_per_paid_gpu_equivalent": baseline["performance"]["aggregate_tok_s_per_paid_gpu_equivalent"],
            "best_aggregate_tok_s_per_paid_gpu_equivalent": best["aggregate_tok_s_per_paid_gpu_equivalent"],
            "throughput_per_paid_gpu_speedup": 1.0,
            "cost_per_million_output_tokens_usd": best["cost_per_million_output_tokens_usd"],
            "accepted_tokens_per_target_pass": None,
            "perfect_block7_depth8_upper_bound_tok_s": architecture["perfect_acceptance_upper_bound"]["dependency_bound_tok_s"],
            "best_microcell_depth": 8,
            "best_measured_expert_workers": micro["best_measured"]["workers"],
            "best_measured_microwork_efficiency_percent": 100 * micro["best_measured"]["relative_complete_layer_throughput"],
            "microwork_worker_gib": micro["best_measured"]["worker_gib"],
            "dcp_8k_degree8_shaped_gain_percent": 100 * dcp["best_8k_sensitivity"]["stage_gain_fraction"],
            "paid_gpu_equivalents": best["paid_gpu_equivalents"],
            "nominal_paid_vram_gib_at_24gib_per_stage": 93 * 24,
            "network_bytes_per_token": 92 * activation_payload_bytes(1),
            "synchronization_count_per_token": 92,
            "coarse_synchronization_count_per_token_depth8": 11,
            "ttft_ms": None,
            "inter_token_latency_p50_ms": None,
            "inter_token_latency_p95_ms": None,
            "inter_token_latency_p99_ms": None,
            "gpu_utilization": None,
            "communication_overlap_fraction": None,
            "full_93_layer_cuda_regression": (
                {
                    "status": "PASS",
                    "layers_executed": 93,
                    "routing_equality": True,
                    "stateful_decode_executed": True,
                    "maximum_layer_relative_l2_error": full_graph["correctness"][
                        "maximum_layer_relative_l2_error"
                    ],
                    "relative_error_gate": full_graph["correctness"][
                        "relative_error_gate"
                    ],
                    "wall_seconds": full_graph["timing"]["wall_seconds"],
                    "classification": "MEASURED correctness-only; not capacity evidence",
                }
                if full_graph_pass and full_graph is not None
                else {"status": "NOT_RUN_OR_FAILED"}
            ),
        },
        "economics": {
            "gpu_hourly_price_usd": economics.gpu_hourly_price_usd,
            "output_price_per_million_usd": economics.output_price_per_million_usd,
            "margin_fraction": economics.target_gpu_margin_fraction,
            "break_even_tok_s_per_paid_gpu": economics.break_even_tok_s_per_paid_gpu,
            "margin_tok_s_per_paid_gpu": economics.margin_tok_s_per_paid_gpu,
        },
        "evidence_counts": {
            "measured_complete_architecture_results": 0,
            "validated_model_architecture_results": 0,
            "shaped_architecture_results": 1,
            "unadmitted_projected_sensitivities": 11,
        },
        "evidence_classification_audit": classification_audit,
        "experiment_016_plan_created": False,
    }
    atomic_json(artifact_root / "machine-summary.json", summary)
    atomic_json(
        artifact_root / "final-recommended-architecture.json",
        {
            "schema_version": "experiment-015-final-architecture-v1",
            "status": "NO_CREDIBLE_ARCHITECTURE_RECOMMENDATION",
            "physical_validation_ready": False,
            "control": "B015-000",
            "best_model_sensitivity": best,
            "candidate_for_further_local_work": {
                "microcell_depth": 8,
                "internal_network": {"maximum_rtt_ms": 0.25, "minimum_bandwidth_gbps": 25.0},
                "coarse_network": {"rtt_ms": 5.0, "bandwidth_gbps": 10.0},
                "reason": "minor shaped dependency improvement only",
            },
            "blocking_requirements": [
                "representative target-verified DSpark acceptance",
                "GPU DSpark draft latency and paid-compute accounting",
                "real device-resident EP transport",
                "complete Kimi DCP correctness/performance",
            ],
            "experiment_016_plan": None,
        },
    )
    return summary


def _manager_summary(summary: dict[str, Any], artifact_root: Path) -> str:
    metrics = summary["metrics"]
    routing = read_json(artifact_root / "expert-routing" / "results.json")
    predictor = routing["previous_token_same_layer_predictor"]
    return f"""## Verdict

* Experiment 015: **FAIL** — unresolved implementation/correctness evidence
* Best architecture: no qualifying Experiment 015 architecture; B015-000 remains the only complete control. The best justified sensitivity is eight-layer microcells.
* Speculative decoding: real public DSpark weights produced finite deterministic draft blocks; representative target acceptance and target-distribution equivalence were not established.
* Pipeline-aware speculation: **NOT ESTABLISHED**; the synchronous prerequisite failed and the pinned public runtime rejects DSpark pipeline parallelism.
* Best accepted tokens/target pass: **NOT ESTABLISHED** (public external cross-workload mean: 3.85, not Swarm evidence)
* Best dependency-bound decode: **{metrics['best_dependency_bound_tok_s']:.4f} tok/s** (SHAPED, microcells only)
* Baseline dependency decode: **0.95 tok/s**
* Dependency speedup: **{metrics['dependency_speedup']:.3f}x**
* Best aggregate tok/s/GPU-equivalent: **{metrics['best_aggregate_tok_s_per_paid_gpu_equivalent']:.4f}**
* Baseline: **1.04**
* Throughput/$ speedup: **1.000x**
* Break-even target: **2.78**
* 50%-margin target: **5.56**
* Best microcell depth: **8 layers** (SHAPED, not physical)
* Best expert microwork size: **4 workers**
* Microwork memory/worker: **{metrics['microwork_worker_gib']:.3f} GiB tracked**
* Best microwork layer efficiency: **{metrics['best_measured_microwork_efficiency_percent']:.2f}% MEASURED**
* Minimum microwork network: **<=0.5 ms RTT and >=2.5 Gbps** for the retained Experiment 014 practical domain; the redesigned direct-buffer minimum is not established.
* DCP gain: **{metrics['dcp_8k_degree8_shaped_gain_percent']:.2f}% SHAPED** at 8K/DCP8; no complete Kimi DCP result
* Best DCP degree: **NOT ESTABLISHED** (DCP8 is only the best latency sensitivity)
* Useful TP operations: **LM head only in a SHAPED TP4 sensitivity; none retained**
* Expert load imbalance: **p95 hottest/coldest ratio {routing['static_modulo_placement']['p95_hottest_to_coldest_ratio']:.1f}x** on a non-representative three-token trace
* Expert replication gain: **NOT ESTABLISHED**
* Route-prediction usefulness: **REJECTED**; precision/recall {100*predictor['precision']:.2f}%
* Minimum useful GPU worker memory: **NOT ESTABLISHED**; 0.915 GiB is the smallest measured shard, while 8 GiB is the smallest modeled class with safe room for the promoted four-way ready delta.
* Paid-GPU-equivalent requirement: **{metrics['paid_gpu_equivalents']:.0f}**
* Projected cost/M output at $0.15/hr: **${metrics['cost_per_million_output_tokens_usd']:.2f}**
* >=5 tok/s/user target: **FAIL**
* >=2.78 tok/s/GPU target: **FAIL**
* >=10 tok/s/user stretch: **FAIL**
* >=5.56 tok/s/GPU stretch: **FAIL**
* Ready for physical architecture validation: **NO**
* Full 93-layer Kimi CUDA regression: **PASS** — 93/93 layers, exact routing, stateful decode, maximum relative L2 {metrics['full_93_layer_cuda_regression']['maximum_layer_relative_l2_error']:.3e}

1. **Why was the 93-stage architecture slow?** Every token crosses 93 serial transformer layers, 92 coarse synchronizations, and about 747 ms of compute; aggregate batching cannot remove this dependency path.
2. **How much did speculation help?** It did not produce an admissible Swarm speedup. Even the zero-draft-cost, perfect block-7/depth-8 upper bound is only {metrics['perfect_block7_depth8_upper_bound_tok_s']:.2f} tok/s.
3. **How much did reducing coarse boundaries help?** Depth 8 reduced modeled latency from 1,052.3 to about 849.9 ms, a {100*(metrics['dependency_speedup']-1):.1f}% speed improvement.
4. **Did microworkers become economically useful?** No. Four-way EP remained 78.81% of the one-GPU layer and concurrent compute cost was not reduced by memory packing.
5. **What specifically improved microwork efficiency?** No implementation improvement was retained. Removing individually measured host/serialization overhead yields a 95.56% optimistic ceiling, not a benchmark.
6. **Did DCP fix the MLA context bottleneck?** Not yet. Exact component reduction passed, and an 8K/DCP8 shaped model showed {metrics['dcp_8k_degree8_shaped_gain_percent']:.1f}% stage gain, but full Kimi state/capacity did not run.
7. **Where did TP help?** Only the LM head in a shaped aggressive-network sensitivity; KDA projection collectives outweighed compute saved.
8. **How important was expert imbalance?** Static placement showed material in-fixture imbalance; a fitted plan cut mean critical selections by 10.16%, but the trace was too small for promotion.
9. **Could route prediction hide communication?** The simple predictor could usefully anticipate only 17.09% of predicted expert activations, so no.
10. **Could GPU expert residency be materially reduced?** Not established; three tokens cannot measure a cache hit rate or steady-state H2D cost.
11. **Which combination won?** None. Microcells alone are the only retained model improvement.
12. **What is now the main bottleneck?** Multi-row target verification service across all 93 layers, followed by unresolved resident EP/DCP implementations.
13. **What hardware/network topology does the winning architecture need?** There is no winning architecture. The microcell sensitivity assumes <=0.25 ms/25 Gbps internally and 5 ms/10 Gbps between cells while retaining 93 paid GPU-equivalents.
14. **Does it reach interactive serving?** No: 1.1766 tok/s vs 5 tok/s.
15. **Does it reach commercial break-even at $0.15/GPU-hour?** No: 1.0446 tok/s/GPU-equivalent vs 2.78.
16. **What should Experiment 016 physically validate?** Nothing yet. First complete representative DSpark verification and a real resident EP or DCP kernel locally; only then select hardware from the measured topology.
"""


def _main_report(summary: dict[str, Any], artifact_root: Path) -> str:
    metrics = summary["metrics"]
    return f"""# Experiment 015: Break the Depth Barrier

Experiment 015 is an architecture research failure, not a negative commercial proof. It did not establish a correct, representative speculative Kimi path or implement the distributed EP/DCP/TP mechanisms needed to decide the full search. It did establish several useful upper bounds that prevent an expensive physical canary.

## Research outcome

The answer to the ultimate simultaneous question is **not demonstrated**. The best admitted architecture-level result is an eight-layer microcell **SHAPED** result at {metrics['best_dependency_bound_tok_s']:.4f} dependency-bound tok/s and {metrics['best_aggregate_tok_s_per_paid_gpu_equivalent']:.4f} aggregate tok/s per paid-GPU-equivalent. It misses both primary thresholds.

More strongly, the optimistic zero-draft-cost, perfect-acceptance block-7 plus depth-8 bound is only {metrics['perfect_block7_depth8_upper_bound_tok_s']:.4f} tok/s. That bound proves the currently modeled batch-verification service cannot reach 5 tok/s merely by adding DSpark and reducing slow boundaries.

## Evidence contract

* **MEASURED** means locally executed real Kimi weights and real CUDA. There are no exceptions.
* **SHAPED** means measured payloads/timings under synthetic transport.
* **VALIDATED MODEL** means a model validated against held-out real execution; the component service model reached {read_json(artifact_root / 'model-validation' / 'results.json')['median_absolute_percentage_error_percent']:.2f}% median error. Results that add synthetic topology remain **SHAPED**.
* **PROJECTED** means an architecture or economic consequence of those inputs.

CPU correctness diagnostics are marked `scientific_result: false` and have no evidence class.

External DSpark, vLLM, DeepEP, EPLB, DCP, and offloading claims appear only in the research-source ledger. They are not Swarm results.

## Immutable B015-000 control

The Experiment 014 capture remains unchanged: 93 stages, one layer per stage, safe batch 8, batch 9 fail-closed, 97.15 aggregate tok/s, 0.9503 dependency-bound tok/s, and 1.0446 aggregate tok/s per paid-GPU-equivalent. At the configured $0.15/GPU-hour this is ${read_json(artifact_root / 'baseline' / 'B015-000.json')['performance']['cost_per_million_output_tokens_usd']:.2f}/M output tokens.

Its dependency path is 747.16 ms compute plus 92 coarse boundaries. At depth 8, the service model exposes only 11 coarse boundaries but retains 81 internal boundaries and every layer's compute.

## Hypothesis cycles

The complete hypothesis → implementation → benchmark → inspection → redesign record is in [cycle-ledger.md](../artifacts/experiment-015/cycle-ledger.md). Failed directions are not hidden.

## Component findings

### Speculation

The pinned 7.12 GB public DSpark checkpoint executed real weights against retained real Kimi hidden states. Blocks 1/2/3/5/7 were finite and deterministic, and greedy/stochastic transaction tests covered full acceptance, partial acceptance, first rejection, full rejection, EOS, cancellation, rollback, repetition, and request isolation.

That is not enough to certify speculative inference. No representative workload received target verification, so accepted length, positional acceptance, target traversals/output token, draft GPU cost, rollback cost, and end-to-end speedup are **NOT ESTABLISHED**. The CPU reference timing is correctness instrumentation, not capacity evidence.

### Pipeline-aware speculation and cancellation

This branch stopped at its prerequisite. The pinned vLLM source explicitly raises for DSpark plus pipeline parallelism, while Swarm lacks a passing synchronous DSpark baseline. Transaction cancellation had zero corruption in unit tests, but layers/CUDA/messages avoided were not benchmarked.

### Hierarchical microcells

Depth 1/2/4/8 produced 0.9503/1.0668/1.1365/1.1766 tok/s in the shaped topology model. Depth 8 is best within the preregistered sweep. It is a useful 23.8% latency sensitivity, not a depth-barrier solution.

### Expert microwork and overlap

The real Kimi sweep remains 2/4/8/16 workers. Four workers are best, at 3.931 GB tracked per worker and 78.81% relative complete-layer throughput. A zero-replacement-cost subtraction of host copies, serialization, exposed transport, and imbalance gives a 95.56% ceiling. Since no device-resident transport implementation ran, the >=90% research target failed.

BF16 and FP16 were checked only as real-hidden-row transport round trips; their full-layer correctness gates did not run. FP8/MXFP8 is unsupported by the retained Kimi transport kernel. No low-precision format was promoted.

### Decode Context Parallelism

Stable max/denominator/numerator combination reproduced unsharded attention to machine precision for degrees 1/2/4/8. The shaped Kimi timing model predicts an 8K DCP8 stage gain of {metrics['dcp_8k_degree8_shaped_gain_percent']:.2f}%. Complete Kimi MLA state, cancellation/recovery, aggregate capacity, and paid compute did not run, so neither the beneficial context threshold nor optimal degree is established.

### Targeted TP

Real CUDA TP1 operation timings were combined with a shaped 0.25 ms/25 Gbps collective. KDA q/output and the MLA projection group did not beat TP1; LM-head TP4 had a 1.63x operation sensitivity. It saves too little of the 1,052 ms path to justify unmeasured extra workers, so no TP was retained.

### Expert placement, prediction, and residency

The available trace contains only three target tokens across 92 MoE layers. Static modulo placement had a p95 hottest/coldest ratio of 8x. A trace-fitted plan reduced mean critical selections by 10.16% in-sample, which is not held-out evidence. The previous-token predictor achieved only 17.09% precision/recall and was rejected. Replication, dynamic migration, and steady-state tiered residency were stopped for insufficient trace history.

### Packing and economics

Memory packing is modeled independently of compute. The 0.915 GiB 16-way shard is the smallest measured logical worker, but it delivered poor layer efficiency. A 24 GiB device can memory-pack all four 3.931 GB expert shards, yet one device cannot provide four-way concurrent compute. Unknown hardware prices, bandwidth, and compute are never imputed from VRAM.

The best admitted projection still requires 93 paid-GPU-equivalents, about {metrics['nominal_paid_vram_gib_at_24gib_per_stage']} GiB nominal VRAM under the legacy 24 GiB stage assumption, and costs ${metrics['cost_per_million_output_tokens_usd']:.2f}/M output tokens.

## Required serving metrics

| Metric | Best justified value | Evidence |
| --- | ---: | --- |
| Dependency-bound tok/s | {metrics['best_dependency_bound_tok_s']:.4f} | SHAPED |
| Aggregate tok/s / paid-GPU-equivalent | {metrics['best_aggregate_tok_s_per_paid_gpu_equivalent']:.4f} | PROJECTED |
| Cost/M output | ${metrics['cost_per_million_output_tokens_usd']:.2f} | PROJECTED, configurable economics |
| TTFT | NOT ESTABLISHED | no end-to-end serving run |
| p50/p95/p99 inter-token latency | NOT ESTABLISHED | no Experiment 015 end-to-end run |
| Logical EP memory/worker | {metrics['microwork_worker_gib']:.3f} GiB | MEASURED |
| Total active GPU-equivalents | {metrics['paid_gpu_equivalents']:.0f} | PROJECTED |
| Network bytes/token | {metrics['network_bytes_per_token']:,} | PROJECTED FP32 92-boundary payload |
| Synchronizations/token | 92 total; 11 coarse at depth 8 | SHAPED |
| GPU utilization | NOT ESTABLISHED | no physical topology |
| Communication overlap | NOT ESTABLISHED | no resident async implementation |
| Numerical fidelity | control PASS; speculative/DCP full path NOT ESTABLISHED | mixed, explicitly scoped |

## Answers to the 30 hard questions

1. Speculation's Swarm contribution is not measured; the perfect combined upper bound is 2.18 tok/s.
2. Accepted tokens/target traversal on our workloads: **NOT ESTABLISHED**.
3. Async vs ordinary speculation: **NOT ESTABLISHED**.
4. Microcells reduce slow boundaries from 92 to 11 at depth 8 and improve modeled speed 23.8%.
5. The best tested microcell depth is 8.
6. Expert microwork did not improve beyond 78.8% in a measured implementation.
7. Reaching the 95.6% ceiling requires persistent device buffers, direct packed activations, device routing metadata, and async device transport; this remains a hypothesis.
8. Smallest measured shard: 0.915 GiB; smallest economically useful footprint: **NOT ESTABLISHED**.
9. Four expert workers remain best measured.
10. Retained practical network domain: <=0.5 ms RTT and >=2.5 Gbps; redesigned admission is not certified.
11. Speculative batching economics: **NOT ESTABLISHED**; batch-8 verification already consumes aggregate capacity.
12. DCP's full Kimi effect: **NOT ESTABLISHED**; shaped stage sensitivity is positive.
13. DCP break-even context: **NOT ESTABLISHED**.
14. Optimal DCP workers: **NOT ESTABLISHED**.
15. Only the LM head showed a TP sensitivity; none passed a complete-stage gate.
16. KDA/MLA collective overhead outweighed savings by TP2/4 in the modeled internal network; LM head preferred TP4.
17. Real-route skew was material in the three-token fixture, with p95 hottest/coldest ratio 8x.
18. Hot-expert replication: **NOT ESTABLISHED**.
19. Dynamic placement is not justified without representative drift traces.
20. Previous-token route prediction was too weak at 17.09% precision/recall.
21. Removable expert VRAM under steady-state tiering: **NOT ESTABLISHED**.
22. Tiered-residency throughput loss: **NOT ESTABLISHED**.
23. Tiering tok/s/$ benefit: **NOT ESTABLISHED**.
24. Best measured complete architecture: B015-000 control; best justified shaped result: depth-8 cells.
25. No EP/DCP/TP/speculation combination passed admission.
26. Depth-8 sensitivity assumes <=0.25 ms/25 Gbps internal and 5 ms/10 Gbps coarse links.
27. Memory-only packing spans 8/12/16/24/32/48 GiB classes; compute-qualified worker sizes are not established.
28. Best justified dependency prediction: {metrics['best_dependency_bound_tok_s']:.4f} tok/s.
29. Best justified aggregate efficiency: {metrics['best_aggregate_tok_s_per_paid_gpu_equivalent']:.4f} tok/s/GPU-equivalent.
30. Projected GPU cost: ${metrics['cost_per_million_output_tokens_usd']:.2f}/M output tokens at $0.15/GPU-hour.

## Stop decision and Experiment 016

The experiment stops because all unimplemented branches either depend on the failed standard-DSpark gate or cannot enter the Pareto frontier without representative traces and complete Kimi correctness. No GPU fleet was rented, no RTX 3090 canary ran, and shaped data is not described as physical.

No Experiment 016 physical-validation plan is produced. Physical validation would be premature. The next work should remain local and produce representative DSpark acceptance plus at least one real resident EP or DCP implementation.

## Final regression record

The complete repository suite passed 1,113 tests with 13 explicitly gated skips. The new focused Experiment 015 suite passed 22 tests. Ruff and Mypy passed. The exact final Kimi CUDA binary then executed all 93 layers for prefill and stateful decode against the pinned independent oracle: routing equality was exact and maximum layer relative L2 was {metrics['full_93_layer_cuda_regression']['maximum_layer_relative_l2_error']:.3e} under the unchanged {metrics['full_93_layer_cuda_regression']['relative_error_gate']:.1e} gate. Its {metrics['full_93_layer_cuda_regression']['wall_seconds']:.1f}-second streamed duration is correctness-only and is not used as serving capacity evidence.
"""


def _integrity(root: Path, artifact_root: Path, docs: list[Path]) -> None:
    files: list[Path] = []
    for path in artifact_root.rglob("*"):
        if not path.is_file() or path.name == "evidence-integrity.json":
            continue
        try:
            relative = path.relative_to(artifact_root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == "research-cache":
            continue
        files.append(path)
    files.extend(path for path in docs if path.is_file())
    files.extend(
        path
        for path in (
            root / "src" / "swarm_inference" / "experiments" / "experiment_015"
        ).glob("*.py")
        if path.is_file()
    )
    files.extend(
        path
        for path in (
            root / "scripts" / "experiment_015_finalize.py",
            root / "scripts" / "experiment_015_dspark_reference.py",
            root / "tests" / "unit" / "test_experiment_015_models.py",
            root / "tests" / "unit" / "test_experiment_015_speculation.py",
        )
        if path.is_file()
    )
    identities = [file_identity(path, relative_to=root) for path in sorted(set(files))]
    classification_audit = _classification_audit(artifact_root)
    atomic_json(
        artifact_root / "evidence-integrity.json",
        {
            "schema_version": "experiment-015-evidence-integrity-v1",
            "status": classification_audit["status"],
            "hash_algorithm": "SHA-256",
            "classification_audit": classification_audit,
            "files": identities,
            "file_count": len(identities),
            "excluded": [
                "research-cache/vllm-source (source commit pinned separately)",
                "evidence-integrity.json (self-reference)",
            ],
        },
    )


def finalize_experiment(
    repository_root: Path,
    *,
    economics: EconomicsConfig | None = None,
) -> dict[str, Any]:
    """Regenerate every required Experiment 015 decision artifact."""
    root = repository_root.expanduser().resolve()
    artifact_root = root / "artifacts" / "experiment-015"
    config = economics or EconomicsConfig()
    capture_baseline(root, artifact_root / "baseline" / "B015-000.json", economics=config)
    validate_service_model(root, artifact_root / "model-validation" / "results.json")
    route_source = root / "artifacts" / "experiment-014" / "oracle-full-93" / "routes.txt"
    export_route_rows(route_source, artifact_root / "expert-routing" / "route-trace.csv")
    analyze_routes(
        route_source,
        artifact_root / "expert-routing" / "results.json",
        artifact_root / "expert-placement" / "plans.json",
        artifact_root / "expert-residency" / "results.json",
    )
    _speculation_artifacts(root, artifact_root)
    analyze_microwork(
        root,
        artifact_root / "microwork" / "results.json",
        artifact_root / "network-surfaces" / "expert-microwork.json",
    )
    analyze_targeted_tp(root, artifact_root / "tp" / "results.json")
    _dcp_artifacts(root, artifact_root)
    build_packing_model(root, artifact_root / "worker-packing" / "model.json")
    analyze_pipeline_occupancy(root, artifact_root / "pipeline-occupancy" / "results.json")
    search_architectures(
        root,
        artifact_root / "architecture-pareto" / "results.json",
        artifact_root / "economics" / "results.json",
        economics=config,
    )
    _network_and_capacity(root, artifact_root)
    _source_ledger(root, artifact_root / "research-source-ledger.json")
    _cycle_ledger(artifact_root)
    summary = _gates_and_summary(artifact_root, config)
    generate_figures(root, artifact_root / "figures")

    manager_path = root / "docs" / "experiment-015-manager-summary.md"
    report_path = root / "docs" / "experiment-015-break-the-depth-barrier.md"
    atomic_text(manager_path, _manager_summary(summary, artifact_root))
    atomic_text(report_path, _main_report(summary, artifact_root))
    _integrity(root, artifact_root, [manager_path, report_path])
    return summary


__all__ = ["finalize_experiment"]
