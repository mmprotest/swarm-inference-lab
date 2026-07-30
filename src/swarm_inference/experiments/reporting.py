"""Self-contained HTML report rendering."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def render_html_report(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    scaling_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
) -> Path:
    criteria = summary["acceptance_criteria"]
    failed = [item for item in criteria if item["status"] == "FAIL"]
    status = "PASS" if not failed else "FAIL"
    primary = summary["primary_result"]
    baseline = summary["baseline_result"]
    headline = [
        ("Execution mode", summary["execution_mode"]),
        ("PASS or FAIL", status),
        (
            "Primary aggregate verified output tokens/s",
            primary["aggregate_verified_output_tokens_s"],
        ),
        (
            "Baseline aggregate verified output tokens/s",
            baseline["aggregate_verified_output_tokens_s"],
        ),
        ("Node count", primary["node_count"]),
        ("Concurrent request count", primary["concurrent_request_count"]),
        ("Model", summary["model_id"]),
        ("Values", summary["values"]),
        ("Failed acceptance criteria", len(failed)),
    ]
    headline_rows = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{_fmt(value)}</td></tr>" for label, value in headline
    )
    failed_items = (
        "\n".join(
            f"<li><strong>{html.escape(item['name'])}</strong>: {html.escape(item['reason'])}</li>"
            for item in failed
        )
        or "<li>None</li>"
    )
    criteria_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f'<td class="{item["status"].lower()}">{item["status"]}</td>'
        f"<td><code>{html.escape(json.dumps(item['observed'], default=str))}</code></td>"
        f"<td><code>{html.escape(json.dumps(item['required'], default=str))}</code></td>"
        f"<td>{html.escape(item['reason'])}</td>"
        "</tr>"
        for item in criteria
    )
    scaling_table = _table(
        scaling_rows,
        [
            "node_count",
            "concurrent_requests",
            "throughput",
            "throughput_gain",
            "marginal_throughput",
            "homogeneous_scaling_efficiency",
            "capacity_normalised_efficiency",
        ],
    )
    request_table = _table(
        request_rows,
        [
            "request_id",
            "status",
            "verification_state",
            "decode_tokens_s",
            "time_to_first_token_s",
            "end_to_end_s",
            "retry_count",
            "route_changes",
        ],
        maximum_rows=256,
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>swarm-inference-lab report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; color: #17202a; }}
h1, h2 {{ color: #102a43; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: .9rem; }}
th, td {{ border: 1px solid #bcccdc; padding: .45rem; text-align: left; vertical-align: top; }}
th {{ background: #f0f4f8; }}
.pass {{ color: #137333; font-weight: 700; }}
.fail {{ color: #b31412; font-weight: 700; }}
.warning {{ background: #fff3cd; border: 1px solid #f0c36d; padding: 1rem; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
.charts figure {{ margin: 0; }}
.charts img {{ width: 100%; border: 1px solid #bcccdc; }}
code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>swarm-inference-lab experiment report</h1>
<table>{headline_rows}</table>
<div class="warning"><strong>Interpretation:</strong> Aggregate verified throughput is
not single-request generation speed. This run is labelled
<strong>{html.escape(summary["execution_mode"])}</strong>; {html.escape(summary["values"])}
values must not be presented as physical distributed performance.</div>
<h2>Failed acceptance criteria</h2>
<ul>{failed_items}</ul>
<h2>Acceptance criteria</h2>
<table><thead><tr><th>Criterion</th><th>Status</th><th>Observed</th><th>Required</th><th>Interpretation</th></tr></thead>
<tbody>{criteria_rows}</tbody></table>
<h2>Required metrics</h2>
<ul>
<li>Aggregate verified output tokens/s: {_fmt(primary["aggregate_verified_output_tokens_s"])}</li>
<li>Per-request tokens/s: see request table and chart.</li>
<li>Mean time to first token: {_fmt(primary["mean_time_to_first_token_s"])} s.</li>
<li>Mean end-to-end latency: {_fmt(primary["mean_end_to_end_s"])} s.</li>
<li>Minimum stage utilisation: {_fmt(primary["minimum_stage_utilisation"])}.</li>
<li>Network traffic: {_fmt(primary["network_bytes"])} bytes.</li>
<li>Recovery: {_fmt(primary["recovered_route_changes"])} route changes,
{_fmt(primary["replay_bytes"])} replay bytes.</li>
</ul>
<h2>Scaling observations</h2>
{scaling_table}
<h2>Per-request results</h2>
{request_table}
<h2>Charts</h2>
<div class="charts">
{_chart_figures()}
</div>
<h2>Reproducibility</h2>
<p>See <code>config.resolved.yaml</code>, <code>environment.json</code>,
<code>model_manifest.json</code>, and the JSONL metric streams in this directory.
Artifact hashes are recorded in <code>artifact_manifest.json</code>.</p>
</body>
</html>
"""
    output = run_dir / "report.html"
    output.write_text(content, encoding="utf-8")
    return output


def _table(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    maximum_rows: int | None = None,
) -> str:
    visible = rows[:maximum_rows] if maximum_rows else rows
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(column, ''))}</td>" for column in columns) + "</tr>"
        for row in visible
    )
    if maximum_rows is not None and len(rows) > maximum_rows:
        body += (
            f'<tr><td colspan="{len(columns)}">Truncated to {maximum_rows} of '
            f"{len(rows)} rows; requests.jsonl contains all rows.</td></tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _chart_figures() -> str:
    names = [
        ("aggregate_throughput.png", "Aggregate verified throughput"),
        ("per_request_throughput.png", "Per-request throughput"),
        ("scaling_efficiency.png", "Scaling efficiency"),
        ("stage_utilisation.png", "Stage utilisation"),
        ("queue_depth.png", "Queue depth"),
        ("network_bytes.png", "Network traffic"),
        ("failure_recovery.png", "Failure and recovery"),
    ]
    return "\n".join(
        f'<figure><img src="charts/{filename}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for filename, caption in names
    )
