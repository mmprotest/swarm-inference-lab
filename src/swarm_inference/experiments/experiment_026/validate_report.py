"""Render and mechanically QA the user-required E026 Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
import re

from markdown_it import MarkdownIt

from . import EXPERIMENT_ID
from .io import file_digest, utc_now, write_once


ROOT = Path("artifacts/experiment-026")
REPORT = Path("docs/experiments/EXPERIMENT_026_REPORT.md")
NOTES = ROOT / "report-source-notes.json"
EXPECTED_SECTIONS = [
    "1. Canonical verdict",
    "2. Executive result",
    "3. Best configuration",
    "4. Performance table",
    "5. Bottleneck decomposition",
    "6. Synchronization result",
    "7. Compression result",
    "8. Cold-start result",
    "9. Churn result",
    "10. Correctness",
    "11. Cost",
    "12. Negative results",
    "13. Scientific interpretation",
    "14. Next decision",
]


def main() -> None:
    source = REPORT.read_text(encoding="utf-8")
    notes = json.loads(NOTES.read_text(encoding="utf-8"))
    tokens = MarkdownIt("commonmark").enable("table").parse(source)
    html = MarkdownIt("commonmark").enable("table").render(source)
    h1 = [tokens[index + 1].content for index, token in enumerate(tokens[:-1]) if token.type == "heading_open" and token.tag == "h1"]
    h2 = [tokens[index + 1].content for index, token in enumerate(tokens[:-1]) if token.type == "heading_open" and token.tag == "h2"]
    table_count = sum(token.type == "table_open" for token in tokens)
    verdict_lines = re.findall(r"^`([A-Z_]+)`$", source, flags=re.MULTILINE)
    checks = [
        {"name": "single_plain_english_title", "pass": h1 == ["E026: Qwen3.8-27B Q4 WAN Swarm Integrated Proof"], "details": h1},
        {"name": "required_14_sections_in_order", "pass": h2 == EXPECTED_SECTIONS, "details": h2},
        {"name": "canonical_verdict_exact", "pass": "`WAN_SWARM_NOT_VIABLE_UNDER_TESTED_CONDITIONS`" in source, "details": None},
        {"name": "direct_answer_first", "pass": "No. E026 did not prove practical WAN swarm inference" in source[:700], "details": None},
        {"name": "audit_tables_rendered", "pass": table_count == 4 and html.count("<table>") == 4, "details": {"tables": table_count}},
        {"name": "major_gate_values_present", "pass": all(value in source for value in (
            "1.615", "0.590", "282 / 512", "9.508", "103.19x", "$0.918195", "178", "98.33%")), "details": None},
        {"name": "exact_and_inexact_labeled", "pass": "fastest exact WAN target run" in source and "rejected inexact MTP result" in source, "details": None},
        {"name": "required_negative_and_limitations", "pass": all(value in source for value in (
            "No point-in-time checkpoint file was transferred", "exclude TCP/IP and SSH framing",
            "No retry or post-seal tuning", "not a universal impossibility proof")), "details": None},
        {"name": "single_decisive_recommendation", "pass": source.count("Continue solving a specific primitive") == 1
            and "Do not proceed to volunteer-runtime engineering" in source, "details": None},
        {"name": "source_notes_match_report", "pass": notes["reporting_job"]["delivery_artifact"] == str(REPORT).replace("\\", "/")
            and notes["reporting_job"]["audience"] == "technical", "details": notes["reporting_job"]},
        {"name": "markdown_structure_closed", "pass": source.count("```") % 2 == 0 and "<table>" not in source, "details": None},
        {"name": "one_primary_verdict_line", "pass": verdict_lines == ["WAN_SWARM_NOT_VIABLE_UNDER_TESTED_CONDITIONS"],
         "details": verdict_lines},
    ]
    failures = [check for check in checks if not check["pass"]]
    output = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": utc_now(),
        "report": str(REPORT),
        "report_sha256": file_digest(REPORT),
        "rendering": {"engine": "markdown-it-py CommonMark with table rule", "html_bytes": len(html.encode("utf-8")),
                      "h1_count": len(h1), "h2_count": len(h2), "table_count": table_count},
        "assessment": "READY_TO_SHARE" if not failures else "NEEDS_REVISION",
        "checks": checks,
        "check_summary": {"passed": len(checks) - len(failures), "failed": len(failures), "total": len(checks)},
        "failures": failures,
        "rendered_html_retained": False,
        "rendered_html_omission_reason": "The user explicitly required Markdown; HTML was rendered in memory only for structural QA, not delivered as a second report surface.",
    }
    path = ROOT / "validation" / "final-report-v2.json"
    write_once(path, output)
    print(json.dumps({"assessment": output["assessment"], **output["check_summary"],
                      "report_sha256": output["report_sha256"], "validation": str(path)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
