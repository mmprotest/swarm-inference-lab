"""Reconcile Experiment 017 evidence and emit its durable artifact package."""

# The report deliberately uses typographic multiplication signs.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from swarm_inference.experiments.experiment_017 import BLOCK_SIZES

SCHEMA_VERSION = "experiment-017-final-analysis-v1"
E015_TOK_S = 2.1838203220693897
E015_MS = 3663.3050435299524
E016_TOK_S = 2.6691235692169797
E016_TRACKER_TOK_S = 2.6691
E016_MS = 2997.238528880436
E016_KDA_MS = 1559.369291174753
E016_MLA_MS = 1195.3071639298962
E016_ENDPOINT_MS = 58.244246095787275
E016_TOPOLOGY_MS = 165.44697167999993
E016_DCP_MS = 18.870856
E016_NON_KDA_MS = E016_MS - E016_KDA_MS
GOAL_TOK_S = 5.0
STRONG_GOAL_TOK_S = 5.3382
GOAL_MS_PER_ACCEPTED = 200.0
USER_SLOTS = 97.15028433444262 / E015_TOK_S
PAID_GPU_EQUIVALENTS = 93.0
GPU_HOURLY_PRICE_USD = 0.15


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            args,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def _source(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(root))
    except ValueError:
        display = str(resolved)
    return {
        "path": display.replace("\\", "/"),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _performance(receipt: dict[str, Any], block: int) -> dict[str, Any]:
    return next(
        row
        for row in receipt["layers"]["89"]["performance_rows"]
        if int(row["candidate_count"]) == block
        and row["arm"] == "C_verification_major_expert_batching"
    )


def _tracker_row(
    *,
    name: str,
    block: int,
    accepted: int,
    target_ms: float,
    kda_ms: float,
    largest_remaining_ms: float,
    exactness: str,
    retained: bool,
) -> dict[str, Any]:
    oracle = accepted * 1000.0 / target_ms
    non_kda = target_ms - kda_ms
    allowed_total = accepted * GOAL_MS_PER_ACCEPTED
    required_kda = allowed_total - non_kda
    required_additional = kda_ms / required_kda if required_kda > 0 else None
    free_kda = accepted * 1000.0 / non_kda if non_kda > 0 else math.inf
    free_largest = accepted * 1000.0 / (target_ms - largest_remaining_ms)
    return {
        "name": name,
        "exactness": exactness,
        "retained": retained,
        "block_size": block,
        "accepted_tokens": accepted,
        "target_pass_ms": target_ms,
        "ms_per_accepted_token": target_ms / accepted,
        "oracle_tok_s_per_user": oracle,
        "required_speedup_vs_016": GOAL_TOK_S / E016_TRACKER_TOK_S,
        "observed_speedup_vs_016": oracle / E016_TRACKER_TOK_S,
        "remaining_gap_tok_s": max(0.0, GOAL_TOK_S - oracle),
        "remaining_gap_ms_per_accepted": max(0.0, target_ms / accepted - GOAL_MS_PER_ACCEPTED),
        "measured_kda_ms": kda_ms,
        "measured_non_kda_ms": non_kda,
        "required_kda_ms_if_non_kda_frozen": required_kda,
        "required_additional_kda_speedup": required_additional,
        "free_kda_oracle": free_kda,
        "largest_remaining_component_ms": largest_remaining_ms,
        "free_largest_remaining_component_oracle": free_largest,
    }


def _economic_row(
    name: str,
    tok_s: float,
    *,
    evidence: str,
    reference_tok_s: float,
) -> dict[str, Any]:
    aggregate = tok_s * USER_SLOTS
    gpu_hours = PAID_GPU_EQUIVALENTS * 1_000_000.0 / (aggregate * 3600.0)
    cost = gpu_hours * GPU_HOURLY_PRICE_USD
    reference_aggregate = reference_tok_s * USER_SLOTS
    reference_gpu_hours = PAID_GPU_EQUIVALENTS * 1_000_000.0 / (reference_aggregate * 3600.0)
    return {
        "architecture": name,
        "evidence_class": evidence,
        "tok_s_per_user": tok_s,
        "retained_user_slots": USER_SLOTS,
        "aggregate_tok_s": aggregate,
        "gpu_equivalent_count": PAID_GPU_EQUIVALENTS,
        "gpu_hours_per_1m_output_tokens": gpu_hours,
        "gpu_hourly_price_usd_inherited": GPU_HOURLY_PRICE_USD,
        "projected_infrastructure_usd_per_1m_output_tokens": cost,
        "change_vs_reference_cost_percent": 100.0 * (gpu_hours / reference_gpu_hours - 1.0),
        "scope": (
            "projection using Experiment 015's retained user slots, 93 paid GPU "
            "equivalents, and $0.15/GPU-hour; target-only zero-draft capacity"
        ),
    }


def _model_metadata(root: Path) -> dict[str, Any]:
    inherited = _read(root / "artifacts" / "experiment-016" / "model-metadata.json")
    inherited["schema_version"] = "experiment-017-kimi-k3-metadata-v1"
    inherited["identity_reused_from"] = "Experiments 014-016"
    inherited["identity_reverified"] = bool(
        inherited["config"]["sha256"] == _sha256(Path(inherited["config"]["path"]))
        and inherited["weight_index"]["sha256"] == _sha256(Path(inherited["weight_index"]["path"]))
    )
    return inherited


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _report(analysis: dict[str, Any]) -> str:
    final = analysis["primary_result"]
    baseline = analysis["baseline_reproduction"]
    arm_a = next(row for row in analysis["arms"] if row["arm_id"] == "A")
    arm_c = next(row for row in analysis["arms"] if row["arm_id"] == "C")
    arm_e = next(row for row in analysis["arms"] if row["arm_id"] == "E")
    tests = analysis["test_results"]
    block_rows = analysis["exact_combined"]
    block_table = _markdown_table(
        [
            "Block",
            "Accepted",
            "Target ms",
            "ms/accepted",
            "tok/s/user",
            "vs E016",
            "KDA ms",
            "KDA budget",
        ],
        [
            [
                str(row["block_size"]),
                str(row["accepted_tokens"]),
                f"{row['target_pass_ms']:.2f}",
                f"{row['ms_per_accepted_token']:.2f}",
                f"{row['oracle_tok_s_per_user']:.4f}",
                f"{row['speedup_vs_exp016']:.3f}×",
                f"{row['kda_contribution_ms']:.2f}",
                f"{row['required_kda_ms_if_non_kda_frozen']:.2f}",
            ]
            for row in block_rows
        ],
    )
    economics = _markdown_table(
        ["Architecture", "tok/s/user", "Aggregate tok/s", "GPU-h/1M", "$/1M"],
        [
            [
                row["architecture"],
                f"{row['tok_s_per_user']:.4f}",
                f"{row['aggregate_tok_s']:.2f}",
                f"{row['gpu_hours_per_1m_output_tokens']:.2f}",
                f"{row['projected_infrastructure_usd_per_1m_output_tokens']:.2f}",
            ]
            for row in analysis["economics"]
        ],
    )
    return f"""# Experiment 017: Breaking the KDA Target-Work Wall

**FAIL — exact.** The best exact zero-draft oracle is **{final["oracle_tok_s_per_user"]:.4f} tok/s/user** at block {final["block_size"]} (**{final["ms_per_accepted_token"]:.2f} ms/accepted token**), {final["speedup_vs_exp016"]:.3f}× versus the fixed Experiment 016 block-7 denominator and {final["speedup_vs_exp015"]:.3f}× versus Experiment 015. It does **not** cross 5 tok/s/user. No Experiment 017 mechanism survived its whole-layer gate: the one-launch exact KDA window was marginally slower, direct accepted-state seeding was already present, exact factors reduced hypothetical snapshot capacity but not target wall time, replay added work, and the already-retained fused native MXFP4 expert path remained fastest. The decisive finding is that the recurrent KDA core itself is only a small fraction of the real layer; the original 1559.4 ms KDA allocation is not 1559.4 ms of state-update work.

> **The block-7 KDA budget to reach 5 tok/s is 162.16 ms. That requires a 9.62× reduction from the declared 1559.4 ms KDA allocation if the declared 1437.84 ms non-KDA work is frozen.**

![Headline oracle progress](../../artifacts/experiment-017/charts/chart-01-oracle-progress.png)

## 1. Hypothesis

The hypothesis was that short-window KDA execution, compact exact state factors, eliminated acceptance copies, replay, and faster native expert service could reorganize Kimi K3 verification enough to reach 5 tok/s/user without changing the model, drafter, routing, topology, or DCP assumptions. Each arm followed hypothesis → implementation → benchmark → inspection → redesign. Exact and approximate evidence remained separate.

The implementation study used [FlashKDA](https://github.com/MoonshotAI/FlashKDA), [its design deep dive](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md), [vLLM's Kimi K3 fused-decode description](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-27-k3.md), [SGLang's KDA fusion release history](https://github.com/sgl-project/sglang/releases), [SpecLA](https://arxiv.org/abs/2607.16673), [Bole](https://arxiv.org/abs/2608.01651), and [Snakes and Ladders / Activation Replay](https://proceedings.mlr.press/v262/wu24a.html) as design references, not as transferable performance claims.

## 2. Hard targets

- PASS: ≥5.0000 tok/s/user, ≤200.0 ms/accepted token, and ≤1600.0 ms for block 7.
- PASS_STRONG: ≥5.3382 tok/s/user and ≤187.325 ms/accepted token.
- Fixed denominators: E015 = {E015_TOK_S:.10f} tok/s; E016 = {E016_TRACKER_TOK_S:.4f} tok/s.
- Initial block-7 budget: 1600.0 − 1437.84 = 162.16 ms of KDA, or 9.62× below 1559.4 ms.

The machine-readable calculation is in `artifacts/experiment-017/target-tracker.json` and is recalculated for every combined block.

## 3. Baseline reproduction

The immutable historical E016 result remains the denominator. The current block-7 reconstruction was **{baseline["current_target_pass_ms"]:.2f} ms**, **{baseline["current_ms_per_accepted"]:.2f} ms/accepted**, and **{baseline["current_oracle_tok_s_per_user"]:.4f} tok/s/user**, a {baseline["target_pass_deviation_percent"]:.2f}% latency deviation from 2997.24 ms. This passes the ±3% gate. Layer 89 was {baseline["layer89_device_deviation_percent"]:.2f}% slower and the 8K MLA control {baseline["mla8k_device_deviation_percent"]:.2f}% slower than their immutable receipts. Historical and current results are both preserved; the current run did not establish a new denominator.

The physical fixture used real layer-89 and layer-91 weights, all 896 layer experts resident, and the three immutable full-graph boundaries from Experiment 014. The complete graph/oracle and topology remain the same conservative validated-model bridge as Experiment 016; this is not a physical 93-device or WAN measurement.

## 4. KDA mathematical analysis

For one head with state `H[key,value]`, the repository update is

`H_t = D_t H_(t-1) + β_t k_t (v_t − k_tᵀ D_t H_(t-1))ᵀ`.

Therefore `H_t = A_t H_(t-1) + C_t`, with `A_t = D_t − β_t k_t(D_t k_t)ᵀ` and `C_t = β_t k_t v_tᵀ`. Each transition is diagonal plus rank one; prefixes compose exactly, while transition and additive ranks grow by one per token. Candidate output `q_iᵀH_i` can be evaluated from the committed `H_0` and prefix factors without writing every `H_i`. The accepted state can be materialized once.

The exact float32 reference passed blocks 1/2/4/7/12/16. At block 7, output relative L2 was {arm_c["block7_output_relative_l2"]:.3e}, state relative L2 was {arm_c["block7_state_relative_l2"]:.3e}, and the factor ranks were 7. Compact KDA-specific token factors occupy {arm_c["block7_compact_factor_bytes"] / 2**20:.3f} MiB versus {arm_c["block7_full_snapshots_bytes"] / 2**20:.3f} MiB for seven full snapshots, a {arm_c["block7_snapshot_compression_ratio"]:.2f}× capacity reduction.

The wall-time result is the opposite: accepted reconstruction makes the analytical factor path about {arm_c["block7_factor_flop_ratio"]:.3f}× the serial recurrence's scalar work, and the reference factor path was {1 / arm_c["block7_cpu_factorized_speedup"]:.2f}× slower. More importantly, the measured native recurrent core is only 0.04313 ms/token. Eliminating all eight block-7 core calls would cap real layer-89 speedup near {arm_c["free_core_layer_speedup_ceiling"]:.3f}×. C4 CUDA was therefore skipped under the experiment's own progression rule; a factor CUDA kernel could not plausibly close the system gap.

## 5. Short-window kernel results

Arm A compared two exact SM120 organizations on real layer 89:

- split: batched token-parallel projections plus one head-parallel recurrence launch per row;
- fused window: the same projections plus one head-parallel launch that advances every contiguous row.

At block 7, fused attention/pre-MoE took {arm_a["fused_attention_device_ms"]:.4f} ms versus {arm_a["split_attention_device_ms"]:.4f} ms; full layer device time was {arm_a["fused_layer_device_ms"]:.4f} versus {arm_a["split_layer_device_ms"]:.4f} ms. The output, state, routes, and active prefix were bit-identical, but the full layer regressed {arm_a["full_layer_regression_percent"]:.3f}%. The recurrence launch count fell from 8 to 1; the saved launches were not visible at the whole-layer gate. CUDA Graph replay is not implemented by this native ABI, so the report records it as unsupported rather than substituting eager timing.

Nsight Compute 2025.3.1 was found and automated, but NVIDIA denied hardware-counter access with `ERR_NVGPUCTRPERM`. No DRAM, L2, occupancy, tensor-core, register, or shared-memory counter is fabricated. CUDA-event, CPU-wall, memory, logical launch, and traffic evidence remains available.

## 6. State-materialization results

Arm B's premise was already true in the retained zero-draft verifier: candidate rows share one session, recurrence advances the session state in place, and the all-accepted final state directly seeds the next round. Acceptance performs **0 bytes**, **0 copy/scatter launches**, and **0 ms** of accepted-state copying. A full KDA recurrent state is 6,291,456 bytes; convolution windows bring session KDA state to 6,881,280 bytes. The kernel's logical block-7 state loops read 88,080,384 bytes and write 88,080,384 bytes, but this is a source-level traffic estimate, not an HBM counter. Removing a copy that does not exist cannot improve the oracle.

![State traffic](../../artifacts/experiment-017/charts/chart-04-state-traffic.png)

## 7. Factorized verification results

Arm C is retained as a correct mathematical and memory-capacity reference, not as a latency path. Prefix composition, state reconstruction, partial acceptance, continued state, inactive slots, aliasing, repeated invocation, and every required block passed. The factor method saves hypothetical speculative snapshots; the retained zero-draft path never created those snapshots. Its full-state application and rank-growing work prevent a credible wall-time advantage at these short windows.

## 8. Replay/checkpoint results

Arm D exactly reproduced serial states for every block. Replay needs one committed checkpoint plus the compact factor/input buffer and repeats the recurrence after acceptance. Against a hypothetical snapshot-per-candidate design it saves capacity; against the retained in-place zero-draft path it adds {arm_c["block7_compact_factor_bytes"] / 2**20:.3f} MiB and replay compute while saving no hot-path copy. This is a memory-capacity result only and was rejected as a latency solution.

## 9. Expert-service results

The available canonical backend consumes the checkpoint's native MXFP4 packed weights and uint8 scales directly, uploads them once, keeps all 896 active-layer experts resident, and uses float32 stage activations. Real block-7 routing had 128 assignments across 36 touched experts, mean M={arm_e["mean_assignments_per_touched_expert"]:.3f}, maximum M={arm_e["maximum_assignments_for_one_expert"]}, and {arm_e["native_routed_expert_calls"]} native expert calls.

Colibri's fused native MXFP4 gate/up path took {arm_e["fused_routed_expert_ms"]:.4f} ms for routed experts and {arm_e["fused_layer_device_ms"]:.4f} ms for the full layer. The exact unfused control took {arm_e["unfused_routed_expert_ms"]:.4f} and {arm_e["unfused_layer_device_ms"]:.4f} ms respectively. Fusion improved routed service {arm_e["routed_expert_speedup"]:.3f}× and is already the Experiment 016 control; it is not a new Experiment 017 gain. FlashInfer, Marlin, SGLang, and vLLM were not installed as compatible Windows/SM120/K3 execution primitives, so no unsupported large-M benchmark was promoted into Kimi evidence.

![Expert service](../../artifacts/experiment-017/charts/chart-05-expert-service.png)

## 10. Precision matrix

The exact float32 control passed. A BF16 stored state with FP32 update arithmetic was evaluated only after the exact control. At block 16 its output relative L2 was {analysis["approximate_result"]["output_relative_l2"]:.6f}, active-state relative L2 {analysis["approximate_result"]["active_state_relative_l2"]:.6f}, and final hidden relative L2 {analysis["approximate_result"]["final_hidden_relative_l2"]:.6f}; the output already exceeds the full-graph 0.003 qualification tolerance. There was no GPU implementation or ≥1.5× whole-oracle result, so the expensive 93-layer qualification was correctly not rerun. FP8 projection/activation, MXFP4+BF16 activation, and MXFP4+FP8 activation modes are explicit unsupported/fail-closed configuration cells, not silent fallbacks. Result: **APPROX_FAIL**, with no qualified approximate oracle.

![Speed-quality Pareto](../../artifacts/experiment-017/charts/chart-06-speed-quality-pareto.png)

## 11. Exact combined result

No Experiment 017 arm survived its whole-layer latency gate, so the exact combined path retains Experiment 016 unchanged. The required block sweep is:

{block_table}

Block 16 maximizes the zero-draft oracle at {final["oracle_tok_s_per_user"]:.4f} tok/s/user, but this is the already-known E016 block curve—not a new mechanism—and it remains below even the 1.25× threshold. At the primary block-7 control, the exact oracle is still {E016_TOK_S:.4f} tok/s/user.

![Block-size oracle](../../artifacts/experiment-017/charts/chart-07-block-size-oracle.png)

## 12. Approximate combined result

There is no approximate combined system result. The only implemented approximate reference failed the layer-level numerical screen and had no GPU speed evidence. It is not assigned throughput and is not mixed with the exact curve.

## 13. Correctness

- Real fused-window and split layer-89 outputs: bit-identical; exact routes and active state.
- Exact factor reference: relative L2 ≤2e-5 for outputs and states at all required blocks.
- Replay and accepted-state reconstruction: exact within the same reassociation threshold.
- Real expert fused/unfused paths: bit-identical outputs, states, and routes.
- Full 93-layer Experiment 014 serial oracle remains the qualification anchor. It was not rerun for a precision arm that failed before the ≥1.5× gate.
- Targeted regressions: {tests["targeted"]["passed"]} passed, {tests["targeted"]["skipped"]} skipped, {tests["targeted"]["failures"] + tests["targeted"]["errors"]} failed.
- Full repository: {tests["full_repository"]["passed"]} passed, {tests["full_repository"]["skipped"]} skipped, {tests["full_repository"]["failures"] + tests["full_repository"]["errors"]} failed. Exact counts and commands are recorded in `test-results.json`.

## 14. New bottleneck decomposition

Because no arm was retained, the reconciled block-7 decomposition remains KDA {E016_KDA_MS:.1f} ms, MLA {E016_MLA_MS:.1f} ms, endpoint {E016_ENDPOINT_MS:.1f} ms, topology {E016_TOPOLOGY_MS:.1f} ms, and shaped DCP communication {E016_DCP_MS:.1f} ms. The more useful cross-layer phase decomposition is expert compute 1177.6 ms (39.29%) and attention/pre-MoE 1083.0 ms (36.13%). Neither component's zero-cost bound reaches 5 alone. The state recurrence is not the KDA allocation.

![KDA budget](../../artifacts/experiment-017/charts/chart-02-kda-budget-to-5.png)

![Layer decomposition](../../artifacts/experiment-017/charts/chart-03-kda-layer-decomposition.png)

## 15. Zero-draft oracle

The primary question is answered negatively. The best exact curve peaks at {final["oracle_tok_s_per_user"]:.4f}; block 7 remains {E016_TOK_S:.4f}. Draft work can only add cost, so speculative proposal, acceptance, and drafter tuning were not run.

## 16. Economic projection

No PASS or STRONG_PARTIAL architecture exists, so there is no qualifying new commercial architecture. For continuity, the table applies the repository's declared $0.15/GPU-hour, 93-GPU-equivalent, {USER_SLOTS:.3f}-user target-only capacity model. These are projections, not physical bills:

{economics}

The apparent E017 row is block-16 amortization already present in E016. It must not be interpreted as a newly achieved architecture.

## 17. What failed

- Maximum recurrence fusion removed launches but not layer wall time.
- Accepted-state copy elimination had no work to remove.
- Exact DPLR factors were mathematically valid but targeted hypothetical state snapshots, while accepted reconstruction restored the arithmetic cost.
- Replay saved hypothetical capacity but added compute to the retained in-place path.
- The native fused expert control was already optimized; the unfused control regressed.
- BF16 recurrent storage exceeded the numerical screen before any system-speed claim.
- Nsight counters were blocked by host permissions; the failed profiled timings are excluded.

## 18. What was retained

The default exact runtime remains the Experiment 016 verification-major path with fused native MXFP4 expert service. The exact short-window CUDA primitive is available only behind explicit `verification-major-kda-window` capability selection and fails closed if its native export is absent; it is not preferred. Precision modes round-trip explicitly and fail closed when unsupported. The factor/replay code remains a deterministic reference and capacity model.

## 19. Implications for Swarm Inference

Experiment 016's aggregate “KDA” allocation was a layer-family attribution, not evidence that recurrent-state traffic dominated. The real core measurement, the exact factor cost, and the full-layer fusion result jointly falsify the thesis that state handling alone can remove roughly 1.4 seconds from the bridged target pass. Fine-grained speculative work cannot repair a target-only path below 5 tok/s/user.

## 20. Recommendation for Experiment 018

Do not continue factor/replay KDA latency work. If Experiment 018 is run, its precondition should be a supportable SM120 tensor-core primitive for the real small-M native-MXFP4 K3 projection and expert shapes, and its gate must combine measured projection and MoE service through the same whole-verifier bridge. The measured phase roofline says both expert compute and attention/pre-MoE must move; optimizing either alone has a zero-cost bound below 5. If that backend is unavailable, stop rather than construct another scalar-kernel projection.

## Final decision

**NO, KDA OPTIMIZATION PATH FALSIFIED.**

## Artifact index

- Final analysis and tracker: `artifacts/experiment-017/summary.json`, `target-tracker.json`
- Raw physical evidence: `artifacts/experiment-017/physical/`
- Flat results: `artifacts/experiment-017/results/`
- Candidate binary and hashes: `artifacts/experiment-017/cuda/`
- Commands, environment, model identity, sources, tests, seeds, failures, and independent audit: `artifacts/experiment-017/`
"""


def analyze(root: Path) -> dict[str, Any]:
    artifact_root = root / "artifacts" / "experiment-017"
    e014_root = root / "artifacts" / "experiment-014"
    e015_root = root / "artifacts" / "experiment-015"
    e016_root = root / "artifacts" / "experiment-016"

    e015_pareto = _read(e015_root / "architecture-pareto" / "results.json")
    e016 = _read(e016_root / "summary.json")
    e016_oracle_rows = list(e016["oracle_by_block"])
    current = _read(artifact_root / "physical" / "baseline-reproduction.json")
    current_dcp = _read(artifact_root / "physical" / "baseline-dcp-reproduction.json")
    split = _read(artifact_root / "physical" / "kda-split-control-physical.json")
    fused = _read(artifact_root / "physical" / "kda-short-window-physical.json")
    unfused = _read(artifact_root / "physical" / "expert-unfused-control-physical.json")
    factor = _read(artifact_root / "physical" / "factor-reference.json")

    cell = e015_pareto["perfect_acceptance_upper_bound"]["cell"]
    old_kda_stage = float(
        _read(e015_root / "model-validation" / "results.json")["held_out_predictions"][0][
            "calibration_points"
        ]["8"]
    )
    old_mla_stage = float(
        _read(e014_root / "performance" / "h014-034d-contextual-batch-layer91-8k.json")["batches"][
            "8"
        ]["retained"]["device"]["p50_ms"]
    )
    current_kda_stage = float(_performance(current, 7)["device"]["p50_ms"])
    current_dcp8 = next(
        row
        for row in current_dcp["rows"]
        if int(row["context_tokens"]) == 8192 and int(row["degree"]) == 8
    )
    fixed_ms = float(cell["endpoint_ms"]) + (E015_MS - float(cell["compute_ms"]))
    current_target = (
        float(cell["kda_ms"]) * current_kda_stage / old_kda_stage
        + float(cell["mla_ms"]) * float(current_dcp8["device"]["p50_ms"]) / old_mla_stage
        + fixed_ms
        + E016_DCP_MS
    )
    historical_physical = _read(e016_root / "physical" / "verification-major-final.json")
    historical_kda_stage = float(_performance(historical_physical, 7)["device"]["p50_ms"])
    historical_mla_stage = float(
        next(
            row
            for row in _read(e016_root / "dcp" / "gpu-results.json")["rows"]
            if int(row["context_tokens"]) == 8192 and int(row["degree"]) == 8
        )["device"]["p50_ms"]
    )
    current_mla_deviation = 100.0 * (
        float(current_dcp8["device"]["p50_ms"]) / historical_mla_stage - 1.0
    )

    baseline_rows = [
        {
            "baseline": "Experiment 015 canonical",
            "status": "HISTORICAL_IMMUTABLE",
            "block_size": 7,
            "accepted_tokens": 8,
            "target_pass_ms": E015_MS,
            "ms_per_accepted_token": E015_MS / 8,
            "oracle_tok_s_per_user": E015_TOK_S,
            "headline_denominator": False,
        },
        {
            "baseline": "Experiment 016 retained",
            "status": "HISTORICAL_IMMUTABLE",
            "block_size": 7,
            "accepted_tokens": 8,
            "target_pass_ms": E016_MS,
            "ms_per_accepted_token": E016_MS / 8,
            "oracle_tok_s_per_user": E016_TOK_S,
            "headline_denominator": True,
        },
        {
            "baseline": "Experiment 017 current reproduction",
            "status": "PASS_WITHIN_3_PERCENT",
            "block_size": 7,
            "accepted_tokens": 8,
            "target_pass_ms": current_target,
            "ms_per_accepted_token": current_target / 8,
            "oracle_tok_s_per_user": 8000.0 / current_target,
            "headline_denominator": False,
            "deviation_vs_historical_exp016_percent": 100.0 * (current_target / E016_MS - 1.0),
        },
    ]

    short_window_rows: list[dict[str, Any]] = []
    oracle_candidate_rows: list[dict[str, Any]] = []
    exact_combined: list[dict[str, Any]] = []
    tracker_rows: list[dict[str, Any]] = []
    for historical in e016_oracle_rows:
        block = int(historical["candidate_block_size"])
        accepted = int(historical["accepted_tokens"])
        historical_target = float(historical["target_pass_ms"])
        split_row = _performance(split, block)
        fused_row = _performance(fused, block)
        split_device = float(split_row["device"]["p50_ms"])
        fused_device = float(fused_row["device"]["p50_ms"])
        kda_contribution = (
            E016_KDA_MS * split_device / float(_performance(split, 7)["device"]["p50_ms"])
        )
        # Scale block-specific KDA to preserve the historical total curve. This
        # is an attribution within the already-validated bridge, not a new bridge.
        if block == 7:
            kda_contribution = E016_KDA_MS
        elif kda_contribution >= historical_target:
            kda_contribution = historical_target * (E016_KDA_MS / E016_MS)
        non_kda = historical_target - kda_contribution
        mla_estimate = min(
            non_kda,
            E016_MLA_MS * accepted / 8.0 * (historical_target / E016_MS) / (accepted / 8.0),
        )
        allowed_total = accepted * GOAL_MS_PER_ACCEPTED
        required_kda = allowed_total - non_kda
        exact_row = {
            "configuration": "retained Experiment 016 exact path",
            "block_size": block,
            "accepted_tokens": accepted,
            "target_pass_ms": historical_target,
            "ms_per_accepted_token": historical_target / accepted,
            "oracle_tok_s_per_user": accepted * 1000.0 / historical_target,
            "speedup_vs_exp016": (accepted * 1000.0 / historical_target) / E016_TRACKER_TOK_S,
            "speedup_vs_exp015": (accepted * 1000.0 / historical_target) / E015_TOK_S,
            "kda_contribution_ms": kda_contribution,
            "non_kda_contribution_ms": non_kda,
            "required_kda_ms_if_non_kda_frozen": required_kda,
            "required_additional_kda_speedup": (
                kda_contribution / required_kda if required_kda > 0 else None
            ),
            "distance_to_5_tok_s": GOAL_TOK_S - accepted * 1000.0 / historical_target,
            "evidence_class": historical["evidence_class"],
            "new_exp017_mechanism": False,
        }
        exact_combined.append(exact_row)
        candidate_kda = kda_contribution * fused_device / split_device
        candidate_target = non_kda + candidate_kda
        for configuration, target_ms, kda_ms, retained in (
            ("E016 exact control", historical_target, kda_contribution, True),
            ("Arm A fused KDA window", candidate_target, candidate_kda, False),
        ):
            oracle_candidate_rows.append(
                {
                    "configuration": configuration,
                    "exactness": "EXACT",
                    "retained": retained,
                    "block_size": block,
                    "accepted_tokens": accepted,
                    "target_pass_ms": target_ms,
                    "ms_per_accepted_token": target_ms / accepted,
                    "oracle_tok_s_per_user": accepted * 1000.0 / target_ms,
                    "speedup_vs_exp016": (accepted * 1000.0 / target_ms) / E016_TRACKER_TOK_S,
                    "kda_contribution_ms": kda_ms,
                    "non_kda_contribution_ms": target_ms - kda_ms,
                }
            )
        traffic = next(
            row["traffic"] for row in factor["rows"] if int(row["block_tokens"]) == block
        )
        short_window_rows.extend(
            [
                {
                    "organization": "split-projection-head-recurrence",
                    "block_size": block,
                    "verification_rows": accepted,
                    "eager_wall_p50_ms": float(split_row["wall"]["p50_ms"]),
                    "eager_device_p50_ms": split_device,
                    "cuda_graph_replay_ms": None,
                    "cuda_graph_status": "UNSUPPORTED_BY_NATIVE_ABI",
                    "full_real_kda_layer_device_ms": split_device,
                    "attention_pre_moe_device_ms": float(
                        split_row["phase_decomposition"]["attention_and_pre_moe"]["device"][
                            "p50_ms"
                        ]
                    ),
                    "kda_recurrence_kernel_count": accepted,
                    "logical_state_read_bytes": traffic["serial_logical_state_read_bytes"],
                    "logical_state_write_bytes": traffic["serial_logical_state_write_bytes"],
                    "allocated_scratch_bytes": 0,
                    "peak_vram_bytes": split["layers"]["89"]["load"]["resident_device_bytes"],
                    "relative_l2_error": 0.0,
                    "route_exact": True,
                    "state_exact": True,
                    "whole_oracle_tok_s_per_user": accepted * 1000.0 / historical_target,
                },
                {
                    "organization": "fused-short-window-head-recurrence",
                    "block_size": block,
                    "verification_rows": accepted,
                    "eager_wall_p50_ms": float(fused_row["wall"]["p50_ms"]),
                    "eager_device_p50_ms": fused_device,
                    "cuda_graph_replay_ms": None,
                    "cuda_graph_status": "UNSUPPORTED_BY_NATIVE_ABI",
                    "full_real_kda_layer_device_ms": fused_device,
                    "attention_pre_moe_device_ms": float(
                        fused_row["phase_decomposition"]["attention_and_pre_moe"]["device"][
                            "p50_ms"
                        ]
                    ),
                    "kda_recurrence_kernel_count": 1,
                    "logical_state_read_bytes": traffic["serial_logical_state_read_bytes"],
                    "logical_state_write_bytes": traffic["serial_logical_state_write_bytes"],
                    "allocated_scratch_bytes": 0,
                    "peak_vram_bytes": fused["layers"]["89"]["load"]["resident_device_bytes"],
                    "relative_l2_error": 0.0,
                    "route_exact": True,
                    "state_exact": True,
                    "whole_oracle_tok_s_per_user": accepted * 1000.0 / candidate_target,
                },
            ]
        )
        tracker_rows.append(
            _tracker_row(
                name=f"exact_combined_block_{block}",
                block=block,
                accepted=accepted,
                target_ms=historical_target,
                kda_ms=kda_contribution,
                largest_remaining_ms=mla_estimate,
                exactness="EXACT",
                retained=True,
            )
        )

    best_exact = max(exact_combined, key=lambda row: row["oracle_tok_s_per_user"])
    block7_split = _performance(split, 7)
    block7_fused = _performance(fused, 7)
    block7_unfused = _performance(unfused, 7)
    factor7 = next(row for row in factor["rows"] if int(row["block_tokens"]) == 7)

    state_traffic_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for row in factor["rows"]:
        traffic = row["traffic"]
        block = int(row["block_tokens"])
        state_traffic_rows.append(
            {
                "block_size": block,
                "full_recurrent_state_bytes": traffic["full_state_bytes"],
                "logical_state_read_bytes": traffic["serial_logical_state_read_bytes"],
                "logical_state_write_bytes": traffic["serial_logical_state_write_bytes"],
                "accepted_state_copy_bytes": traffic["accepted_state_copy_bytes"],
                "accepted_state_copy_launches": traffic["accepted_state_copy_launches"],
                "accepted_state_copy_wall_ms": 0.0,
                "compact_factor_bytes": traffic["compact_token_factor_bytes"],
                "hypothetical_full_snapshot_bytes": traffic["speculative_full_snapshot_bytes"],
                "full_kda_layer_device_ms": float(_performance(split, block)["device"]["p50_ms"]),
                "traffic_measurement": (
                    "logical bytes from exact kernel/state geometry; no HBM counter"
                ),
            }
        )
        factor_rows.append(
            {
                "block_size": block,
                "status": row["status"],
                "transition_rank": row["transition_rank"],
                "additive_rank": row["additive_rank"],
                "serial_cpu_p50_ms_one_head": row["serial_cpu_timing"]["p50_ms"],
                "factorized_cpu_p50_ms_one_head": row["factorized_cpu_timing"]["p50_ms"],
                "cpu_factorized_speedup": row["cpu_factorized_speedup"],
                "output_max_abs_error": row["output_metrics"]["maximum_absolute_error"],
                "output_relative_l2_error": row["output_metrics"]["relative_l2_error"],
                "state_max_abs_error": row["state_metrics"]["maximum_absolute_error"],
                "state_relative_l2_error": row["state_metrics"]["relative_l2_error"],
                "compact_factor_bytes": traffic["compact_token_factor_bytes"],
                "full_snapshot_bytes": traffic["speculative_full_snapshot_bytes"],
                "snapshot_compression_ratio": traffic["snapshot_compression_ratio"],
                "factor_flop_ratio_vs_serial": row["operation_model"][
                    "factor_flop_ratio_vs_serial"
                ],
                "cuda_prototype": "SKIPPED_BY_COST_GATE",
                "whole_oracle_tok_s_per_user": None,
                "decision": "MEMORY_REFERENCE_ONLY_REJECT_LATENCY",
            }
        )
        replay_rows.append(
            {
                "block_size": block,
                "checkpoint_frequency_tokens": block,
                "replay_cpu_p50_ms_one_head": row["replay_cpu_timing"]["p50_ms"],
                "state_hbm_traffic_saved_vs_hypothetical_snapshots_bytes": max(
                    0,
                    traffic["speculative_full_snapshot_bytes"] - traffic["full_state_bytes"],
                ),
                "extra_compact_buffer_bytes": traffic["compact_token_factor_bytes"],
                "vram_reduction_vs_hypothetical_snapshots_bytes": max(
                    0,
                    traffic["speculative_full_snapshot_bytes"]
                    - traffic["compact_token_factor_bytes"],
                ),
                "vram_change_vs_retained_in_place_path_bytes": traffic[
                    "compact_token_factor_bytes"
                ],
                "total_kda_layer_latency_ms": None,
                "decision": "CAPACITY_ONLY_REJECT_LATENCY",
            }
        )

    expert_rows: list[dict[str, Any]] = []
    for backend, receipt, available, fused_gate_up, note in (
        (
            "Colibri native MXFP4 fused gate/up",
            split,
            True,
            True,
            "canonical; native packed MXFP4 and uint8 scales; repack/upload once",
        ),
        (
            "Colibri native MXFP4 unfused exact control",
            unfused,
            True,
            False,
            "same native representation; separate gate/up kernels",
        ),
    ):
        row = _performance(receipt, 7)
        expert_rows.append(
            {
                "backend": backend,
                "available": available,
                "sm120_supported": True,
                "actual_k3_shape_supported": True,
                "native_checkpoint_representation": True,
                "weight_precision": "MXFP4 group-32",
                "activation_precision": "FP32",
                "preprocessing": "one-time checkpoint load/upload",
                "hot_path_repacking": False,
                "real_small_m_supported": True,
                "verification_major_supported": True,
                "fused_gate_up": fused_gate_up,
                "block7_mean_m": row["routing"]["last_block"][
                    "mean_assignments_per_touched_expert"
                ],
                "block7_max_m": row["routing"]["last_block"]["maximum_assignments_for_one_expert"],
                "expert_phase_device_ms": row["phase_decomposition"]["routed_expert_compute"][
                    "device"
                ]["p50_ms"],
                "full_kda_layer_device_ms": row["device"]["p50_ms"],
                "full_kda_layer_wall_ms": row["wall"]["p50_ms"],
                "correctness_pass": receipt["correctness_pass"],
                "note": note,
            }
        )
    for backend, reason in (
        (
            "FlashInfer native MXFP4",
            "not installed; K3 small-M Windows/SM120 compatibility unestablished",
        ),
        ("Marlin", "not installed; no supportable native K3 MXFP4 shape path in this environment"),
        ("SGLang execution primitive", "not installed; runtime replacement is out of scope"),
        (
            "vLLM K3 fused expert primitive",
            "not installed/exported as a Colibri-compatible Windows primitive",
        ),
    ):
        expert_rows.append(
            {
                "backend": backend,
                "available": False,
                "sm120_supported": None,
                "actual_k3_shape_supported": None,
                "native_checkpoint_representation": None,
                "weight_precision": None,
                "activation_precision": None,
                "preprocessing": None,
                "hot_path_repacking": None,
                "real_small_m_supported": None,
                "verification_major_supported": None,
                "fused_gate_up": None,
                "block7_mean_m": None,
                "block7_max_m": None,
                "expert_phase_device_ms": None,
                "full_kda_layer_device_ms": None,
                "full_kda_layer_wall_ms": None,
                "correctness_pass": None,
                "note": reason,
            }
        )

    precision = factor["precision_reference"]
    precision_rows = [
        {
            "mode": "exact-fp32",
            "classification": "EXACT_CONTROL",
            "support_status": "MEASURED_GPU",
            "exact_control_latency_ms": E016_MS,
            "candidate_latency_ms": E016_MS,
            "speedup": 1.0,
            "max_absolute_error": 0.0,
            "relative_l2_error": 0.0,
            "route_agreement": 1.0,
            "active_state_divergence_relative_l2": 0.0,
            "final_hidden_divergence_relative_l2": 0.0,
            "logit_divergence": 0.0,
            "top1_token_agreement": True,
            "nan_count": 0,
            "inf_count": 0,
            "oracle_tok_s_per_user": E016_TOK_S,
            "qualification": "PASS",
        },
        {
            "mode": precision["mode"],
            "classification": "APPROXIMATE",
            "support_status": "REFERENCE_ONLY_NO_GPU_KERNEL",
            "exact_control_latency_ms": None,
            "candidate_latency_ms": None,
            "speedup": None,
            "max_absolute_error": precision["output_metrics"]["maximum_absolute_error"],
            "relative_l2_error": precision["output_metrics"]["relative_l2_error"],
            "route_agreement": None,
            "active_state_divergence_relative_l2": precision["active_state_metrics"][
                "relative_l2_error"
            ],
            "final_hidden_divergence_relative_l2": precision["final_hidden_metrics"][
                "relative_l2_error"
            ],
            "logit_divergence": None,
            "top1_token_agreement": None,
            "nan_count": precision["output_metrics"]["nan_count"],
            "inf_count": precision["output_metrics"]["inf_count"],
            "oracle_tok_s_per_user": None,
            "qualification": "APPROX_FAIL_LAYER_SCREEN",
        },
    ]
    for mode, reason in (
        ("fp8-projection-activation", "no compatible native projection kernel"),
        ("mxfp4-bf16-activation", "native expert ABI accepts FP32 activations only"),
        ("mxfp4-fp8-activation", "no compatible native expert activation path"),
        ("bf16-state-plus-faster-expert-activation", "both required GPU primitives unavailable"),
        ("lower-precision-recurrent-state", "not pursued after BF16 failed and no speed path"),
    ):
        precision_rows.append(
            {
                "mode": mode,
                "classification": "APPROXIMATE",
                "support_status": "UNSUPPORTED_FAIL_CLOSED",
                "qualification": "NOT_RUN",
                "note": reason,
            }
        )

    best_oracle = float(best_exact["oracle_tok_s_per_user"])
    outcome = "FAIL"
    approximate_outcome = "APPROX_FAIL"
    economics = [
        _economic_row(
            "Experiment 015 block 7",
            E015_TOK_S,
            evidence="validated-model bridge / projection",
            reference_tok_s=E015_TOK_S,
        ),
        _economic_row(
            "Experiment 016 block 7",
            E016_TOK_S,
            evidence="validated-model bridge / projection",
            reference_tok_s=E015_TOK_S,
        ),
        _economic_row(
            "Experiment 017 best exact (existing E016 block-16 curve)",
            best_oracle,
            evidence="validated-model bridge / projection; no new mechanism",
            reference_tok_s=E015_TOK_S,
        ),
    ]
    for row in economics:
        row["change_vs_exp016_cost_percent"] = 100.0 * (
            row["gpu_hours_per_1m_output_tokens"] / economics[1]["gpu_hours_per_1m_output_tokens"]
            - 1.0
        )
        row["change_vs_exp015_cost_percent"] = 100.0 * (
            row["gpu_hours_per_1m_output_tokens"] / economics[0]["gpu_hours_per_1m_output_tokens"]
            - 1.0
        )

    tracker = {
        "schema_version": "experiment-017-target-tracker-v1",
        "baseline_exp015_tok_s": 2.1838203221,
        "baseline_exp016_tok_s": 2.6691,
        "goal_tok_s": 5.0,
        "strong_goal_tok_s": 5.3382,
        "baseline_exp016_ms_per_accepted": 374.65,
        "goal_ms_per_accepted": 200.0,
        "baseline_exp016_block7_ms": 2997.24,
        "goal_block7_ms": 1600.0,
        "baseline_exp016_kda_ms": 1559.4,
        "baseline_exp016_non_kda_ms": 1437.84,
        "initial_required_kda_ms_if_non_kda_frozen": 162.16,
        "initial_required_additional_kda_speedup": 1559.4 / 162.16,
        "results": tracker_rows,
    }

    model_metadata = _model_metadata(root)
    test_results_path = artifact_root / "test-results.json"
    test_results = _read(test_results_path)
    candidate_dll = artifact_root / "cuda" / "coli_cuda-sm120-h017-window.dll"
    historical_dll = e016_root / "cuda" / "coli_cuda-sm120-h016-final.dll"
    nvcuda = Path("C:/Windows/System32/nvcuda.dll")
    cuda_hashes = [_source(root, historical_dll), _source(root, candidate_dll)]
    if nvcuda.exists():
        cuda_hashes.append(_source(root, nvcuda))
    ncu_candidates = sorted(
        Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute */ncu.bat")
    )
    nsight_compute: dict[str, Any]
    if ncu_candidates:
        ncu_path = ncu_candidates[-1]
        nsight_compute = {
            "status": "AVAILABLE_COUNTER_PERMISSION_DENIED",
            "path": str(ncu_path),
            "version": ncu_path.parent.name.removeprefix("Nsight Compute "),
            "decisive_profile_attempt": "ERR_NVGPUCTRPERM",
        }
    else:
        nsight_compute = {"status": "UNAVAILABLE"}
    environment = {
        "schema_version": "experiment-017-environment-v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host_ram_total_bytes": psutil.virtual_memory().total,
        "git_commit": _command(["git", "rev-parse", "HEAD"], cwd=root),
        "git_branch": _command(["git", "branch", "--show-current"], cwd=root),
        "git_status": _command(["git", "status", "--short"], cwd=root),
        "colibri_commit": _command(
            [
                "git",
                "-c",
                f"safe.directory={(root / 'third_party' / 'colibri').resolve()!s}",
                "rev-parse",
                "HEAD",
            ],
            cwd=root / "third_party" / "colibri",
        ),
        "colibri_status": _command(
            [
                "git",
                "-c",
                f"safe.directory={(root / 'third_party' / 'colibri').resolve()!s}",
                "status",
                "--short",
            ],
            cwd=root / "third_party" / "colibri",
        ),
        "gpu": _command(["nvidia-smi"]),
        "cuda_compiler": _command(["nvcc", "--version"]),
        "nsight_compute": nsight_compute,
        "nsight_systems": "UNAVAILABLE",
        "cuda_dll_hashes": cuda_hashes,
        "test_status": {
            "status": test_results["status"],
            "targeted": test_results["targeted"],
            "full_repository": test_results["full_repository"],
            "receipt": "artifacts/experiment-017/test-results.json",
        },
    }

    manifest_paths = [
        e016_root / "summary.json",
        e016_root / "physical" / "verification-major-final.json",
        e016_root / "physical" / "mla-8k-block-sweep-final.json",
        e016_root / "dcp" / "gpu-results.json",
        e014_root / "oracle-full-93" / "serial-oracle-receipt.json",
        e014_root / "oracle-full-93" / "hidden-trace.f32",
        e014_root / "cuda" / "h014-025y-real-kda-core.json",
        artifact_root / "physical" / "baseline-reproduction.json",
        artifact_root / "physical" / "baseline-mla8k-reproduction.json",
        artifact_root / "physical" / "baseline-dcp-reproduction.json",
        artifact_root / "physical" / "kda-split-control-physical.json",
        artifact_root / "physical" / "kda-short-window-physical.json",
        artifact_root / "physical" / "expert-unfused-control-physical.json",
        artifact_root / "physical" / "factor-reference.json",
        test_results_path,
        artifact_root / "targeted-pytest.xml",
        artifact_root / "full-pytest.xml",
        candidate_dll,
        root / "src" / "swarm_inference" / "execution" / "kda_verification.py",
        root
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_017"
        / "reference_benchmark.py",
        root / "src" / "swarm_inference" / "experiments" / "experiment_017" / "finalize.py",
        root / "src" / "swarm_inference" / "experiments" / "experiment_017" / "figures.py",
        root / "src" / "swarm_inference" / "experiments" / "experiment_017" / "test_receipt.py",
        root / "src" / "swarm_inference" / "experiments" / "experiment_017" / "audit.py",
        root / "tests" / "test_kda_verification.py",
    ]
    source_manifest = {
        "schema_version": "experiment-017-source-manifest-v1",
        "files": [_source(root, path) for path in manifest_paths],
    }

    arm_rows = [
        {
            "arm_id": "A",
            "name": "exact short-window KDA verification",
            "hypothesis": "one recurrent launch per window reduces real layer wall time",
            "implementation": "SM120 one-block-per-head kernel loops contiguous rows",
            "benchmark": "real Kimi K3 layer 89, paired DLL, blocks 1/2/4/7/12/16",
            "result": "bit exact; block-7 full layer regressed",
            "redesign": "reject; launch overhead is not the wall",
            "retained": False,
            "split_attention_device_ms": block7_split["phase_decomposition"][
                "attention_and_pre_moe"
            ]["device"]["p50_ms"],
            "fused_attention_device_ms": block7_fused["phase_decomposition"][
                "attention_and_pre_moe"
            ]["device"]["p50_ms"],
            "split_layer_device_ms": block7_split["device"]["p50_ms"],
            "fused_layer_device_ms": block7_fused["device"]["p50_ms"],
            "full_layer_regression_percent": 100.0
            * (block7_fused["device"]["p50_ms"] / block7_split["device"]["p50_ms"] - 1),
        },
        {
            "arm_id": "B",
            "name": "eliminate accepted-state copy/scatter",
            "hypothesis": "acceptance copies a full recurrent state",
            "implementation": "inspect and instrument retained in-place state progression",
            "benchmark": "block sweep logical traffic + real layer wall",
            "result": "zero accepted-state bytes, launches, and wall time already",
            "redesign": "no-op control retained; no new gain",
            "retained": False,
        },
        {
            "arm_id": "C",
            "name": "KDA-specific factorized verification",
            "hypothesis": "DPLR prefixes avoid full speculative states and reduce wall time",
            "implementation": "exact affine diagonal-plus-low-rank reference",
            "benchmark": "C1-C3 exact block sweep and CPU/GPU cost model",
            "result": "correct and 42.56x snapshot compression; no plausible latency gain",
            "redesign": "skip C4 CUDA under cost gate; retain memory reference only",
            "retained": False,
            "block7_output_relative_l2": factor7["output_metrics"]["relative_l2_error"],
            "block7_state_relative_l2": factor7["state_metrics"]["relative_l2_error"],
            "block7_compact_factor_bytes": factor7["traffic"]["compact_token_factor_bytes"],
            "block7_full_snapshots_bytes": factor7["traffic"]["speculative_full_snapshot_bytes"],
            "block7_snapshot_compression_ratio": factor7["traffic"]["snapshot_compression_ratio"],
            "block7_factor_flop_ratio": factor7["operation_model"]["factor_flop_ratio_vs_serial"],
            "block7_cpu_factorized_speedup": factor7["cpu_factorized_speedup"],
            "free_core_layer_speedup_ceiling": factor["cuda_prototype_gate"][
                "free_core_layer_speedup_ceiling"
            ],
        },
        {
            "arm_id": "D",
            "name": "replay/checkpoint state",
            "hypothesis": "compact replay is cheaper than speculative state snapshots",
            "implementation": "exact checkpoint plus compact token factors",
            "benchmark": "exact block sweep and replay CPU/traffic model",
            "result": "capacity win only versus a design the retained oracle does not use",
            "redesign": "reject latency path",
            "retained": False,
        },
        {
            "arm_id": "E",
            "name": "KDA-layer expert service",
            "hypothesis": "native fused MXFP4 improves fragmented small-M service",
            "implementation": "canonical Colibri fused gate/up versus exact unfused control",
            "benchmark": "real layer-89 routes and all 896 resident experts",
            "result": "fused is faster but was already the E016 canonical control",
            "redesign": "retain existing control; no incremental E017 gain",
            "retained": False,
            "mean_assignments_per_touched_expert": block7_split["routing"]["last_block"][
                "mean_assignments_per_touched_expert"
            ],
            "maximum_assignments_for_one_expert": block7_split["routing"]["last_block"][
                "maximum_assignments_for_one_expert"
            ],
            "native_routed_expert_calls": block7_split["routing"]["last_block"][
                "native_routed_expert_calls"
            ],
            "fused_routed_expert_ms": block7_split["phase_decomposition"]["routed_expert_compute"][
                "device"
            ]["p50_ms"],
            "unfused_routed_expert_ms": block7_unfused["phase_decomposition"][
                "routed_expert_compute"
            ]["device"]["p50_ms"],
            "routed_expert_speedup": block7_unfused["phase_decomposition"]["routed_expert_compute"][
                "device"
            ]["p50_ms"]
            / block7_split["phase_decomposition"]["routed_expert_compute"]["device"]["p50_ms"],
            "fused_layer_device_ms": block7_split["device"]["p50_ms"],
            "unfused_layer_device_ms": block7_unfused["device"]["p50_ms"],
        },
    ]

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "verdict": outcome,
        "exactness": "EXACT",
        "final_decision": "NO, KDA OPTIMIZATION PATH FALSIFIED",
        "primary_result": {
            "outcome": outcome,
            "exactness": "EXACT",
            "oracle_tok_s_per_user": best_oracle,
            "ms_per_accepted_token": best_exact["ms_per_accepted_token"],
            "target_pass_ms": best_exact["target_pass_ms"],
            "block_size": best_exact["block_size"],
            "accepted_tokens": best_exact["accepted_tokens"],
            "speedup_vs_exp016": best_oracle / E016_TRACKER_TOK_S,
            "speedup_vs_exp015": best_oracle / E015_TOK_S,
            "crossed_5_tok_s": False,
            "decisive_mechanism": "none; all Experiment 017 latency arms rejected",
            "note": "best block is the pre-existing Experiment 016 block-16 curve",
        },
        "baseline_reproduction": {
            "historical_target_pass_ms": E016_MS,
            "historical_ms_per_accepted": E016_MS / 8,
            "historical_oracle_tok_s_per_user": E016_TOK_S,
            "current_target_pass_ms": current_target,
            "current_ms_per_accepted": current_target / 8,
            "current_oracle_tok_s_per_user": 8000.0 / current_target,
            "target_pass_deviation_percent": 100.0 * (current_target / E016_MS - 1),
            "layer89_device_deviation_percent": 100.0
            * (current_kda_stage / historical_kda_stage - 1),
            "mla8k_device_deviation_percent": current_mla_deviation,
            "gate_percent": 3.0,
            "pass": abs(current_target / E016_MS - 1.0) <= 0.03,
        },
        "arms": arm_rows,
        "exact_combined": exact_combined,
        "approximate_result": {
            "outcome": approximate_outcome,
            "qualified": False,
            "oracle_tok_s_per_user": None,
            "output_relative_l2": precision["output_metrics"]["relative_l2_error"],
            "active_state_relative_l2": precision["active_state_metrics"]["relative_l2_error"],
            "final_hidden_relative_l2": precision["final_hidden_metrics"]["relative_l2_error"],
            "full_93_layer_run": False,
            "reason": "failed layer screen and no >=1.5x whole-oracle GPU result",
        },
        "correctness": {
            "short_window_pass": fused["correctness_pass"],
            "short_window_bit_identity": True,
            "expert_controls_pass": split["correctness_pass"] and unfused["correctness_pass"],
            "factor_reference_pass": factor["status"] == "PASS",
            "exact_relative_l2_gate": 2e-5,
            "approximate_full_graph_relative_l2_gate": 0.003,
            "full_93_layer_anchor": str(
                e014_root / "oracle-full-93" / "serial-oracle-receipt.json"
            ),
        },
        "new_bottleneck": {
            "block7_components_ms": {
                "KDA": E016_KDA_MS,
                "MLA": E016_MLA_MS,
                "endpoint": E016_ENDPOINT_MS,
                "topology_communication": E016_TOPOLOGY_MS,
                "DCP_communication": E016_DCP_MS,
            },
            "phase_components_ms": {
                row["component"]: row["modeled_wall_ms"]
                for row in e016["detailed_final_decomposition"]
            },
            "largest_phase": "expert compute",
            "largest_phase_free_oracle_tok_s": 8000.0 / (E016_MS - 1177.5841990224867),
            "attention_pre_moe_free_oracle_tok_s": 8000.0 / (E016_MS - 1082.9831719139595),
            "single_phase_crosses_5": False,
        },
        "profiling": {
            "cuda_event_timings": True,
            "cpu_wall_timings": True,
            "gpu_memory_samples": True,
            "nsight_compute_available": True,
            "nsight_compute_counter_status": "BLOCKED_ERR_NVGPUCTRPERM",
            "nsight_systems_available": False,
            "hardware_counters_reported": False,
            "failed_ncu_timings_excluded": True,
        },
        "economics": economics,
        "target_tracker": tracker,
        "environment": environment,
        "model_metadata": model_metadata,
        "run_seeds": {
            "reference_math_seed": factor["seed"],
            "physical_gpu_benchmarks": {
                "seed": None,
                "reason": "deterministic replay of three immutable real Kimi boundaries",
                "oracle_trace_sha256": _sha256(e014_root / "oracle-full-93" / "hidden-trace.f32"),
            },
        },
        "source_manifest": source_manifest,
        "test_results": test_results,
        "artifacts": {
            "baseline_rows": baseline_rows,
            "short_window_rows": short_window_rows,
            "state_traffic_rows": state_traffic_rows,
            "factor_rows": factor_rows,
            "replay_rows": replay_rows,
            "expert_rows": expert_rows,
            "precision_rows": precision_rows,
            "oracle_rows": oracle_candidate_rows,
            "economics_rows": economics,
        },
    }
    return analysis


def finalize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    artifact_root = root / "artifacts" / "experiment-017"
    analysis = analyze(root)
    rows = analysis.pop("artifacts")
    _atomic_json(artifact_root / "summary.json", analysis)
    _atomic_json(artifact_root / "target-tracker.json", analysis["target_tracker"])
    _atomic_json(artifact_root / "environment.json", analysis["environment"])
    _atomic_json(artifact_root / "model-metadata.json", analysis["model_metadata"])
    _atomic_json(artifact_root / "run-seeds.json", analysis["run_seeds"])
    _atomic_json(artifact_root / "source-manifest.json", analysis["source_manifest"])

    results = artifact_root / "results"
    _write_csv(results / "baseline.csv", rows["baseline_rows"])
    _write_csv(results / "kda-short-window.csv", rows["short_window_rows"])
    _write_csv(results / "state-traffic.csv", rows["state_traffic_rows"])
    _write_csv(results / "factorized-verification.csv", rows["factor_rows"])
    _write_csv(results / "replay-state.csv", rows["replay_rows"])
    _write_csv(results / "expert-backends.csv", rows["expert_rows"])
    _write_csv(results / "exact-combined.csv", analysis["exact_combined"])
    _write_csv(results / "precision-matrix.csv", rows["precision_rows"])
    _write_csv(results / "oracle-by-block.csv", rows["oracle_rows"])
    _write_csv(results / "economics.csv", rows["economics_rows"])

    physical = artifact_root / "physical"
    split_path = physical / "kda-split-control-physical.json"
    window_path = physical / "kda-short-window-physical.json"
    unfused_path = physical / "expert-unfused-control-physical.json"
    block7_short = [row for row in rows["short_window_rows"] if int(row["block_size"]) == 7]
    real_kda_summary = {
        "schema_version": "experiment-017-real-kda-layer-summary-v1",
        "claim_boundary": {
            "physical": "real Kimi K3 layer 89 on one RTX 5090",
            "model_bridge": "whole-verifier oracle uses the unchanged Experiment 016 validated model",
            "not_claimed": ["physical 93-device result", "physical WAN result"],
        },
        "fixture": {
            "checkpoint": analysis["model_metadata"]["checkpoint"],
            "layer": 89,
            "experts_resident": 896,
            "resident_device_bytes": rows["short_window_rows"][0]["peak_vram_bytes"],
            "block_sizes": list(BLOCK_SIZES),
            "warmup_iterations": 5,
            "retained_iterations": 30,
            "profile_iterations": 5,
        },
        "block7": {
            "split_control": block7_short[0],
            "fused_short_window": block7_short[1],
            "unfused_expert_control": next(
                row
                for row in rows["expert_rows"]
                if row["backend"] == "Colibri native MXFP4 unfused exact control"
            ),
        },
        "correctness": analysis["correctness"],
        "decision": {
            "short_window_retained": False,
            "expert_change_retained": False,
            "reason": "neither candidate improved the exact whole-layer control",
        },
        "raw_receipts": [
            _source(root, split_path),
            _source(root, window_path),
            _source(root, unfused_path),
        ],
    }
    _atomic_json(physical / "real-kda-layer-results.json", real_kda_summary)

    gpu_rows: list[dict[str, Any]] = []
    for sample_path in sorted(physical.glob("gpu-samples-*.csv")):
        with sample_path.open(encoding="utf-8", newline="") as handle:
            for sample in csv.DictReader(handle):
                gpu_rows.append(
                    {
                        "source_run": sample_path.stem.removeprefix("gpu-samples-"),
                        "excluded_from_performance": sample_path.name == "gpu-samples-ncu.csv",
                        **sample,
                    }
                )
    _write_csv(physical / "gpu-samples.csv", gpu_rows)

    cuda_files = sorted(
        path
        for path in (artifact_root / "cuda").iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    _atomic_json(
        artifact_root / "cuda" / "manifest.json",
        {
            "schema_version": "experiment-017-cuda-manifest-v1",
            "files": [_source(root, path) for path in cuda_files],
            "candidate_dll": "coli_cuda-sm120-h017-window.dll",
            "preferred_runtime_path": False,
        },
    )

    failure_log = {
        "schema_version": "experiment-017-failure-log-v1",
        "failures": [
            {
                "stage": "initial CUDA build",
                "status": "RESOLVED",
                "error": "VS 18 Insiders was not returned without vswhere -prerelease",
                "resolution": "build helper now includes -prerelease; ALLOW_UNSUPPORTED=1 used with CUDA 13.0",
                "scientific_impact": "none; paired candidate/control use the same rebuilt DLL",
            },
            {
                "stage": "Nsight Compute hardware counters",
                "status": "UNRESOLVED_ENVIRONMENT_PERMISSION",
                "error": "ERR_NVGPUCTRPERM",
                "resolution": "none; no administrative driver setting changed",
                "scientific_impact": "counter metrics unavailable; failed profiled timings excluded",
            },
            {
                "stage": "CUDA Graph replay",
                "status": "UNSUPPORTED",
                "error": "native Colibri ABI has no graph capture/replay export",
                "resolution": "reported eager CUDA-event timing only",
                "scientific_impact": "no graph-replay claim",
            },
        ],
    }
    _atomic_json(artifact_root / "failure-log.json", failure_log)

    commands = """# Experiment 017 command transcript (PowerShell)
git status --short
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_016.benchmark --checkpoint F:\\models\\Kimi-K3 --cuda-library artifacts\\experiment-016\\cuda\\coli_cuda-sm120-h016-final.dll --oracle-trace artifacts\\experiment-014\\oracle-full-93\\hidden-trace.f32 --output artifacts\\experiment-017\\physical\\baseline-reproduction.json --gpu-samples artifacts\\experiment-017\\physical\\gpu-samples-baseline.csv --layers 89,91 --warmup 5 --iterations 30 --profile-iterations 5
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_016.context_benchmark --checkpoint F:\\models\\Kimi-K3 --cuda-library artifacts\\experiment-016\\cuda\\coli_cuda-sm120-h016-final.dll --oracle-trace artifacts\\experiment-014\\oracle-full-93\\hidden-trace.f32 --output artifacts\\experiment-017\\physical\\baseline-mla8k-reproduction.json --gpu-samples artifacts\\experiment-017\\physical\\gpu-samples-mla8k-baseline.csv --context 8192 --layer 91 --warmup 3 --iterations 20 --profile-iterations 3 --block-sizes 1,2,4,7,12,16
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_016.dcp_cuda_benchmark --checkpoint F:\\models\\Kimi-K3 --cuda-library artifacts\\experiment-016\\cuda\\coli_cuda-sm120-h016-final.dll --oracle-trace artifacts\\experiment-014\\oracle-full-93\\hidden-trace.f32 --output artifacts\\experiment-017\\physical\\baseline-dcp-reproduction.json --contexts 2,8,32 --degrees 1,2,4,8 --warmup 2 --iterations 12 --profile-iterations 2
$env:CUDA_OUTPUT=<artifact-root>\\cuda\\coli_cuda-sm120-h017-window.dll; $env:ALLOW_UNSUPPORTED='1'; third_party\\colibri\\c\\build_cuda.bat
# Paired real layer-89 runs used experiment_016.benchmark with the rebuilt DLL, --warmup 5 --iterations 30 --profile-iterations 5 and respectively:
# --fast-path-mode verification-major
# --fast-path-mode verification-major-kda-window
# --fast-path-mode verification-major --fused-gate-up false
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_017.reference_benchmark --output artifacts\\experiment-017\\physical\\factor-reference.json --warmup 2 --iterations 7
# Nsight attempt: ncu --kernel-name regex:kimi_kda_short_window --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis (failed ERR_NVGPUCTRPERM; timing excluded)
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m pytest -q tests\\test_kda_verification.py tests\\unit\\test_kimi_indexed_copy_runtime.py tests\\unit\\test_kimi_k3_adapter.py --junitxml artifacts\\experiment-017\\targeted-pytest.xml
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m pytest -q --junitxml artifacts\\experiment-017\\full-pytest.xml
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_017.test_receipt --targeted-xml artifacts\\experiment-017\\targeted-pytest.xml --full-xml artifacts\\experiment-017\\full-pytest.xml --output artifacts\\experiment-017\\test-results.json
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_017.finalize --root .
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_017.figures --root .
$env:PYTHONPATH=(Resolve-Path src).Path; .\\.venv\\Scripts\\python.exe -m swarm_inference.experiments.experiment_017.audit --root .
"""
    _atomic_text(artifact_root / "commands.txt", commands)
    _atomic_text(root / "docs" / "experiments" / "EXPERIMENT_017_REPORT.md", _report(analysis))
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    analysis = finalize(arguments.root)
    print(
        json.dumps(
            {
                "verdict": analysis["verdict"],
                "primary_result": analysis["primary_result"],
                "final_decision": analysis["final_decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
