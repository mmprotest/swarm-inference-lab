"""Create audited E026 summary/receipt revisions without mutating retained synthesis artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

from .io import file_digest, utc_now, write_once


ROOT = Path("artifacts/experiment-026")
DATASET = ROOT / "metrics-canonical-v2.jsonl"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]


def row_by_id(records: list[dict], run_id: str) -> dict:
    matches = [row for row in records if row["run_id"] == run_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {run_id!r} record, found {len(matches)}")
    return matches[0]


def main() -> None:
    records = rows()
    direct = row_by_id(records, "wan-direct-cache-pipeline-no-checkpoints-001")
    direct_mtp = row_by_id(records, "wan-direct-cache-pipeline-mtp-k3-001")
    recovery = load(ROOT / "recovery" / "wan-replica-kill-001.json")

    summary = load(ROOT / "summary-canonical.json")
    summary["major_results"].pop("best_wan_tok_s", None)
    summary["major_results"].pop("best_speedup_over_ordinary", None)
    summary["major_results"].update({
        "best_exact_wan_tok_s": direct["decode_tok_s"],
        "best_attempted_inexact_mtp_wan_tok_s": direct_mtp["decode_tok_s"],
        "best_attempted_inexact_mtp_speedup_over_ordinary": (
            direct_mtp["decode_tok_s"] / summary["major_results"]["ordinary_wan_tok_s"]
        ),
    })
    summary["gates"]["interactive_decode"].update({
        "best_exact_tok_s": direct["decode_tok_s"],
        "best_attempted_inexact_mtp_tok_s": direct_mtp["decode_tok_s"],
    })
    summary["synthesis_lineage"].update({
        "canonical_summary": "summary-final.json",
        "canonical_receipt": "final-receipt-canonical-v2.json",
        "supersedes": summary["synthesis_lineage"]["supersedes"]
            + ["summary-canonical.json", "final-receipt-canonical.json"],
        "additional_reason": "correct recovery-resumption timestamp semantics and exact-vs-inexact throughput labels",
    })
    write_once(ROOT / "summary-final.json", summary)

    receipt = load(ROOT / "final-receipt-canonical.json")
    old_recovery_timestamp = receipt["recovery_completion"]["timestamp"]
    injection = datetime.fromisoformat(recovery["failure_injection"]["timestamp"])
    resumed = injection + timedelta(seconds=recovery["user_visible_interruption_s"])
    receipt["recovery_completion"].update({
        "timestamp": resumed.isoformat(),
        "timestamp_basis": "failure injection wall time plus measured token-stream interruption",
        "recovery_receipt_generated_at": old_recovery_timestamp,
    })
    receipt["gates"]["interactive_decode"].update({
        "best_exact_tok_s": direct["decode_tok_s"],
        "best_attempted_inexact_mtp_tok_s": direct_mtp["decode_tok_s"],
    })
    receipt["synthesis_lineage"] = summary["synthesis_lineage"]
    receipt["canonicality"] = {
        "status": "CANONICAL",
        "dataset": str(DATASET),
        "dataset_sha256": file_digest(DATASET),
        "prior_receipts_retained": ["final-receipt.json", "final-receipt-canonical.json"],
    }
    receipt["timestamp"] = utc_now()
    output = ROOT / "final-receipt-canonical-v2.json"
    write_once(output, receipt)
    print(json.dumps({
        "summary": str(ROOT / "summary-final.json"),
        "receipt": str(output),
        "receipt_sha256": file_digest(output),
        "recovery_resumed_at": resumed.isoformat(),
    }), flush=True)


if __name__ == "__main__":
    main()
