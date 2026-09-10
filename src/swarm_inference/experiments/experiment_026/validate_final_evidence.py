"""Reproducible data-quality and analytical validation for canonical E026 evidence."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from . import EXPERIMENT_ID
from .io import file_digest, utc_now, write_once


ROOT = Path("artifacts/experiment-026")
RUNS = ROOT / "runs"
DATASET = ROOT / "metrics-canonical-v3.jsonl"
SUMMARY = ROOT / "summary-canonical-v3.json"
RECEIPT = ROOT / "final-receipt-canonical-v3.json"

REQUIRED_ROW_FIELDS = {
    "experiment_id", "run_id", "timestamp", "repository_commit", "model", "llama_cpp",
    "configuration_hash", "node_identifiers", "gpu_types", "regions", "stage_assignments",
    "pairwise_network", "context_length", "generated_tokens", "ttft_s", "prompt_tok_s",
    "decode_tok_s", "median_tpot_s", "p95_tpot_s", "bytes_sent", "bytes_received",
    "bytes_per_committed_token", "bytes_measurement", "target_traversals",
    "target_traversals_measurement", "wan_traversals", "speculative_proposals",
    "speculative_acceptances", "tokens_per_expensive_target_traversal", "stage_compute_times",
    "stage_network_waits", "pipeline_idle", "gpu_utilization", "state_size_bytes",
    "checkpoint_time_s", "restore_time_s", "failure_timing", "correctness", "cost",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    rows = load_rows()
    summary = load(SUMMARY)
    receipt = load(RECEIPT)
    recovery = load(ROOT / "recovery" / "wan-replica-kill-001.json")
    state = load(ROOT / "state" / "local-hybrid-restore-001" / "receipt.json")
    opening = load(ROOT / "cost" / "opening.json")
    final_provider = load(ROOT / "cost" / "final-provider-snapshot.json")
    collection = load(ROOT / "remote-collected-final-002" / "receipt.json")
    lineage = load(ROOT / "synthesis-lineage.json")
    checks: list[dict] = []

    def check(name: str, passed: bool, details: object) -> None:
        checks.append({"name": name, "pass": bool(passed), "details": details})

    def one(run_id: str) -> dict:
        matches = [row for row in rows if row["run_id"] == run_id]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one {run_id!r} row, found {len(matches)}")
        return matches[0]

    source_paths = sorted(RUNS.glob("*/*/metrics.json"))
    source_rows = [row for row in rows if row.get("source_metrics")]
    check("row_count_and_source_coverage", len(rows) == 71 and len(source_rows) == len(source_paths) == 70,
          {"canonical_rows": len(rows), "source_metric_files": len(source_paths), "derived_failure_rows": 1})

    keys = [(row["run_id"], row["prompt_id"]) for row in rows]
    duplicates = [{"run_id": key[0], "prompt_id": key[1], "count": count}
                  for key, count in Counter(keys).items() if count > 1]
    check("composite_grain_unique", not duplicates,
          {"grain": "one row per run_id and prompt_id", "duplicate_keys": duplicates})

    missing = [{"run_id": row["run_id"], "prompt_id": row["prompt_id"],
                "fields": sorted(REQUIRED_ROW_FIELDS - row.keys())}
               for row in rows if REQUIRED_ROW_FIELDS - row.keys()]
    check("required_schema_complete", not missing, {"missing": missing})

    source_lineage_errors = []
    for row in source_rows:
        source = Path(row["source_metrics"])
        if not source.exists() or source.parent.parent.name != row["run_id"]:
            source_lineage_errors.append({"run_id": row["run_id"], "source": str(source)})
    recorded_mismatches = sum(row.get("source_recorded_run_id") != row["run_id"] for row in source_rows)
    check("canonical_run_id_lineage", not source_lineage_errors and recorded_mismatches == 44,
          {"path_mapping_errors": source_lineage_errors,
           "malformed_legacy_ids_exposed_not_used": recorded_mismatches})

    check("identity_and_evidence_class", all(
        row["experiment_id"] == EXPERIMENT_ID
        and row["repository_commit"] == "d4b4cabd67bb3ef65ce8f2fe997099cb3487822b"
        and row["model"]["sha256"] == "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
        and row["evidence_class"] == "PHYSICAL" for row in rows),
        {"rows": len(rows), "allowed_class": "PHYSICAL"})

    invalid_numeric = []
    for row in rows:
        for field in ("context_length", "generated_tokens", "ttft_s", "prompt_tok_s", "decode_tok_s",
                      "median_tpot_s", "p95_tpot_s", "bytes_sent", "bytes_received", "wan_traversals"):
            value = row.get(field)
            if value is None or not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
                invalid_numeric.append({"run_id": row["run_id"], "prompt_id": row["prompt_id"],
                                        "field": field, "value": value})
    check("core_numeric_domains", not invalid_numeric, {"invalid": invalid_numeric})

    unexplained_bytes = [row["run_id"] for row in rows
                         if row["bytes_per_committed_token"] is None
                         and row["bytes_measurement"] not in {"NOT_APPLICABLE_LOCAL", "UNAVAILABLE_NO_PROTOCOL_TRACE"}]
    unexplained_traversals = [row["run_id"] for row in rows
                              if row["target_traversals"] is None
                              and row["target_traversals_measurement"] != "UNAVAILABLE_PRE_COUNTER_BUILD"]
    check("nulls_are_explicit", not unexplained_bytes and not unexplained_traversals,
          {"unexplained_byte_nulls": unexplained_bytes,
           "unexplained_traversal_nulls": unexplained_traversals,
           "pre_counter_local_mtp_rows": sum(row["target_traversals"] is None for row in rows)})

    wan_rows = [row for row in rows if row["run_id"].startswith("wan-") or row["run_id"] == "final-wan-sealed-001"]
    wan_topology_errors = [row["run_id"] for row in wan_rows
                           if len(row["node_identifiers"]) != 3 or len(row["pairwise_network"]) != 3
                           or row["target_traversals"] is None]
    check("wan_topology_and_traversal_coverage", not wan_topology_errors,
          {"wan_rows": len(wan_rows), "errors": wan_topology_errors})

    remote_hash_errors = []
    remote_size = 0
    for item in collection["files"]:
        path = Path(item["local"])
        actual_size = path.stat().st_size if path.exists() else None
        actual_hash = file_digest(path) if path.exists() else None
        remote_size += actual_size or 0
        if actual_size != item["size_bytes"] or actual_hash != item["sha256"]:
            remote_hash_errors.append({"path": str(path), "expected_size": item["size_bytes"],
                                       "actual_size": actual_size, "expected_sha256": item["sha256"],
                                       "actual_sha256": actual_hash})
    check("remote_collection_integrity", not remote_hash_errors and remote_size == collection["total_bytes"],
          {"files": len(collection["files"]), "bytes": remote_size, "errors": remote_hash_errors})

    canonical_lineage = {item["path"]: item for item in lineage["artifacts"] if item["status"] == "CANONICAL"}
    lineage_errors = []
    for name, item in canonical_lineage.items():
        path = ROOT / name
        if not path.exists() or file_digest(path) != item["sha256"]:
            lineage_errors.append(name)
    check("canonical_synthesis_hashes", not lineage_errors and set(canonical_lineage) == {
        "metrics-canonical-v3.jsonl", "summary-canonical-v3.json", "final-receipt-canonical-v3.json"},
        {"canonical_files": sorted(canonical_lineage), "hash_errors": lineage_errors})

    check("receipt_dataset_binding", receipt["dataset"]["path"] == str(DATASET)
          and receipt["dataset"]["sha256"] == file_digest(DATASET)
          and receipt["dataset"]["rows"] == len(rows), receipt["dataset"])
    check("verdict_consistency", summary["canonical_verdict"] == receipt["canonical_verdict"]
          == "WAN_SWARM_NOT_VIABLE_UNDER_TESTED_CONDITIONS",
          {"summary": summary["canonical_verdict"], "receipt": receipt["canonical_verdict"]})

    ordinary = one("wan-ordinary-profile-001")
    direct = one("wan-direct-cache-pipeline-no-checkpoints-001")
    direct_mtp = one("wan-direct-cache-pipeline-mtp-k3-001")
    proxy_target = one("wan-routed-cache-pipeline-no-checkpoints-001")
    proxy_mtp = one("wan-routed-cache-pipeline-mtp-k3-001")
    final = one("final-wan-sealed-001")
    check("throughput_recomputations", all((
        close(direct["decode_tok_s"] / ordinary["decode_tok_s"], summary["major_results"]["target_only_speedup_over_ordinary"]),
        close(proxy_mtp["decode_tok_s"] / proxy_target["decode_tok_s"], summary["major_results"]["stable_proxy_mtp_speedup"]),
        close(direct_mtp["decode_tok_s"] / ordinary["decode_tok_s"],
              summary["major_results"]["best_attempted_inexact_mtp_speedup_over_ordinary"]),
    )), {"ordinary_tok_s": ordinary["decode_tok_s"], "best_exact_tok_s": direct["decode_tok_s"],
          "best_attempted_inexact_mtp_tok_s": direct_mtp["decode_tok_s"]})

    startup = summary["startup"]
    check("startup_recomputations", all((
        close(startup["standby_same_shard_cold"]["elapsed_s"] /
              startup["standby_same_shard_disk_warm_verify"]["elapsed_s"],
              startup["disk_warm_same_shard_speedup"]),
        close(startup["full_cold_topology_critical_node_s"] /
              startup["best_disk_warm_full_server_ready_s"],
              startup["full_cold_to_best_disk_warm_speedup"]),
    )), {"same_shard_speedup": startup["disk_warm_same_shard_speedup"],
          "scope_warning": "full cold provisioning/build and disk-warm server load are not like-for-like phases"})

    check("hybrid_state_accounting", state["save"]["attention_state_bytes"]
          + state["save"]["recurrent_state_with_header_bytes"] == state["save"]["state_bytes"],
          state["save"])

    check("final_byte_recomputations", all((
        final["coordinator_wan_protocol_bytes"] + final["peer_wan_protocol_bytes"]
            == final["total_wan_protocol_bytes"],
        close(final["total_wan_protocol_bytes"] / final["generated_tokens"],
              final["bytes_per_committed_token_including_startup_all_wan_links"]),
        close(final["median_steady_coordinator_bytes_per_token"]
              + final["median_steady_peer_bytes_per_token"],
              final["median_steady_all_wan_links_bytes_per_token"]),
        final["bytes_per_committed_token"] == final["median_steady_all_wan_links_bytes_per_token"],
    )), {key: final[key] for key in ("generated_tokens", "coordinator_wan_protocol_bytes",
                                     "peer_wan_protocol_bytes", "total_wan_protocol_bytes",
                                     "bytes_per_committed_token_including_startup_all_wan_links",
                                     "median_steady_all_wan_links_bytes_per_token")})

    actual_cost = (opening["budget"]["conservative_available_usd"]
                   - final_provider["budget"]["conservative_available_usd"])
    check("cost_recomputation_and_teardown", close(actual_cost, receipt["cost"]["actual_vast_expenditure_usd"])
          and actual_cost <= 38 and final_provider["all_e026_instances_absent"],
          {"opening_credit_usd": opening["budget"]["conservative_available_usd"],
           "final_credit_usd": final_provider["budget"]["conservative_available_usd"],
           "actual_delta_usd": actual_cost, "ceiling_usd": 38,
           "all_e026_instances_absent": final_provider["all_e026_instances_absent"]})

    resumed = datetime.fromisoformat(recovery["failure_injection"]["timestamp"]).timestamp() \
        + recovery["user_visible_interruption_s"]
    receipt_resumed = datetime.fromisoformat(receipt["recovery_completion"]["timestamp"]).timestamp()
    check("failure_recovery", recovery["status"] == "PASS" and recovery["token_stream_equal_control"]
          and recovery["duplicate_tokens"] == recovery["lost_tokens"] == recovery["full_prompt_replay_tokens"] == 0
          and recovery["user_visible_interruption_s"] < 20 and close(resumed, receipt_resumed, 1e-6),
          {"interruption_s": recovery["user_visible_interruption_s"], "duplicates": recovery["duplicate_tokens"],
           "lost": recovery["lost_tokens"], "full_prompt_replay_tokens": recovery["full_prompt_replay_tokens"],
           "first_resumed_token_timestamp": receipt["recovery_completion"]["timestamp"]})

    mtp_k3 = [row for row in rows if row["run_id"] == "local-mtp-k3-001"]
    mtp_failures = [row for row in mtp_k3 if row["correctness"]["status"] != "PASS"]
    check("correctness_claims", len(mtp_k3) == 6 and len(mtp_failures) == 4
          and final["correctness"]["first_mismatch"] == 178
          and final["correctness"]["status"] == "FAIL_STRICT_GREEDY_AND_INCOMPLETE",
          {"mtp_k3_failed_prompts": len(mtp_failures), "mtp_k3_prompts": len(mtp_k3),
           "sealed_first_mismatch_zero_based": final["correctness"]["first_mismatch"],
           "sealed_completed_tokens": final["generated_tokens"], "sealed_requested_tokens": final["requested_generated_tokens"]})

    failed = [item for item in checks if not item["pass"]]
    limitations = [
        "Twelve early local MTP rows predate the draft-verification counter; the null is explicit and does not affect WAN traversal claims.",
        "Per-run stage timing and GPU-utilization coverage is partial; the sealed final run has aligned worker timing and utilization evidence.",
        "WAN development configurations have one measured prompt each; the result establishes observed behavior on this topology, not a population estimate.",
        "The sealed WAN request is a valid partial physical failure at 282 of 512 tokens, not a completed 512-token demonstration.",
        "Protocol byte counts exclude TCP/IP and SSH framing.",
        "The local coordinator CPU/RAM inventory was captured after the run on the unchanged host and is labeled accordingly.",
    ]
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": utc_now(),
        "question": "Is the canonical E026 evidence accurate and complete enough to support the final architectural decision?",
        "intended_grain": "one PHYSICAL measurement row per run_id and prompt_id, plus one derived row for the valid sealed transport failure",
        "as_of": final_provider["timestamp"],
        "timezone_policy": "source timestamps retain ISO-8601 offsets; cross-source calculations normalize to UTC",
        "overall_assessment": "READY_TO_SHARE" if not failed else "NEEDS_REVISION",
        "checks": checks,
        "check_summary": {"passed": len(checks) - len(failed), "failed": len(failed), "total": len(checks)},
        "blocking_issues": failed,
        "documented_limitations": limitations,
        "visualization_decision": {
            "chart_omitted": True,
            "reason": "The decisive evidence is a sparse set of categorical ablations with exact audit values; a table preserves units, validity status, and non-comparable scopes better than a trend chart.",
        },
        "source_inventory": {
            "dataset": str(DATASET), "dataset_sha256": file_digest(DATASET),
            "summary": str(SUMMARY), "summary_sha256": file_digest(SUMMARY),
            "receipt": str(RECEIPT), "receipt_sha256": file_digest(RECEIPT),
            "raw_metric_files": len(source_paths), "remote_collection_files": len(collection["files"]),
        },
        "recommended_automated_tests": [
            "derive canonical run_id from the immutable directory and assert source-recorded run_id agreement",
            "assert unique (run_id, prompt_id) grain and required top-level schema fields",
            "bind every final receipt to the canonical dataset SHA256",
            "distinguish exact from rejected/inexact performance in gate summaries",
        ],
    }
    output = ROOT / "validation" / "final-evidence-v2.json"
    write_once(output, artifact)
    print(json.dumps({"assessment": artifact["overall_assessment"], **artifact["check_summary"],
                      "path": str(output), "sha256": file_digest(output)}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
