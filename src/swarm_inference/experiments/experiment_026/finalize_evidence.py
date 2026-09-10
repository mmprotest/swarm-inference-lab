"""Build the normalized E026 metrics dataset, summary, and final proof receipt."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

from . import EXPERIMENT_ID
from .analyze_runs import summarize_trace
from .io import digest, file_digest, utc_now, write_once


ROOT = Path("artifacts/experiment-026")
RUNS = ROOT / "runs"
VERDICT = "WAN_SWARM_NOT_VIABLE_UNDER_TESTED_CONDITIONS"
MODEL_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_COMMIT = "f1b6fbf35cfa010b0a8d6301fdfccbb7f41bd903"
REPOSITORY_COMMIT = "d4b4cabd67bb3ef65ce8f2fe997099cb3487822b"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line]


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def write_jsonl_once(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


SEAL = load(ROOT / "seal" / "configuration.json")
DEPLOYMENT = load(ROOT / "services" / "wan-router-services-final-001" / "deployment.json")
NETWORK_A = load(ROOT / "network" / "direct-wan-preflight-001" / "50478198.json")
NETWORK_B = load(ROOT / "network" / "direct-wan-preflight-001" / "50478207.json")
NETWORK_AB = load(ROOT / "network" / "direct-wan-preflight-001" / "peer-a-b.json")
VRAM = load(ROOT / "resource" / "local-vram-post-seal-002" / "receipt.json")
LOCAL_HOST = load(ROOT / "resource" / "local-host-post-run-001.json")
STATE = load(ROOT / "state" / "local-hybrid-restore-001" / "receipt.json")
RECOVERY = load(ROOT / "recovery" / "wan-replica-kill-001.json")
FINAL_PARTIAL = load(ROOT / "analysis" / "final-wan-sealed-001-partial-logits-trace.json")
OPENING = load(ROOT / "cost" / "opening.json")
FINAL_PROVIDER = load(ROOT / "cost" / "final-provider-snapshot.json")
COLLECTION = load(ROOT / "remote-collected-final-002" / "receipt.json")


MODEL = {
    "name": "Qwen3.8-27B",
    "format": "GGUF",
    "quantization": "Q4_K_M",
    "file": "Qwen3.8-27B-Q4_K_M.gguf",
    "size_bytes": 18_973_870_432,
    "sha256": MODEL_SHA,
    "source": "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF",
    "revision": "0669b98607d47046c7c2b3f801011d54a08cfccf",
}

LLAMA = {
    "commit": LLAMA_COMMIT,
    "release_tag": "b10886",
    "version": "0.4.0-dev",
    "instrumentation_diff_sha256": "eed261160912740fb76dc325cc48c2b89cdce136ec915fb615a758312a94c539",
    "local_build": {
        "generator": "Ninja",
        "type": "Release",
        "compiler": "MSVC cl 14.44",
        "flags": ["GGML_CUDA=ON", "GGML_RPC=ON", "CMAKE_CUDA_ARCHITECTURES=120",
                  "LLAMA_BUILD_TESTS=OFF", "LLAMA_BUILD_EXAMPLES=OFF", "LLAMA_OPENSSL=OFF"],
        "cuda": "13.0.88",
        "driver": "591.86",
    },
    "remote_build": {
        "generator": "Ninja",
        "type": "Release",
        "compiler": "GNU 13.3.0",
        "flags": ["GGML_CUDA=ON", "GGML_RPC=ON", "CMAKE_CUDA_ARCHITECTURES=86-real",
                  "LLAMA_BUILD_TESTS=OFF", "LLAMA_BUILD_EXAMPLES=OFF", "LLAMA_BUILD_SERVER=OFF",
                  "LLAMA_OPENSSL=OFF"],
        "cuda": "13.0.88",
        "drivers": {"worker_a": "595.84", "worker_b": "580.159.03"},
    },
}

NODES = [
    {"node_id": "local-rtx5090", "gpu": "NVIDIA GeForce RTX 5090", "vram_mib": 32607,
     "region": "user-local; Australia/Sydney timezone (physical location not independently geolocated)",
     "driver": "591.86", "cpu": LOCAL_HOST["cpu"], "cpu_cores": LOCAL_HOST["cpu_cores"],
     "cpu_logical_processors": LOCAL_HOST["cpu_logical_processors"], "ram_mib": LOCAL_HOST["ram_mib"]},
    {"node_id": 50478198, "machine_id": 140487, "gpu": "RTX 3080 Ti", "vram_mib": 12288,
     "region": "Japan, JP", "driver": "595.84", "cpu": "AMD Ryzen 5 3500 6-Core Processor",
     "ram_mib": 15912, "hourly_usd": 0.17314814814814813},
    {"node_id": 50478207, "machine_id": 29808, "gpu": "RTX 3060", "vram_mib": 12288,
     "region": "South Korea, KR", "driver": "580.159.03", "cpu": "Intel Xeon E5-2680 v4",
     "ram_mib": 64284, "hourly_usd": 0.06305555555555556},
]

STAGES = [
    {"order": 0, "node_id": 50478198, "layers": [0, 8], "cached_tensor_bytes": 2_424_279_168},
    {"order": 1, "node_id": 50478207, "layers": [9, 16], "cached_tensor_bytes": 2_150_593_792},
    {"order": 2, "node_id": "local-rtx5090", "layers": [17, 63],
     "also_owns": ["embedding", "output"], "whole_target_path": False},
]

PAIRWISE = [
    {"source": "local-rtx5090", "destination": 50478198,
     "application_rtt_ms": NETWORK_A["median_rtt_s"] * 1000,
     "p95_rtt_ms": NETWORK_A["p95_rtt_s"] * 1000,
     "jitter_stddev_ms": NETWORK_A["jitter_stddev_s"] * 1000,
     "throughput_mbps": NETWORK_A["throughput"], "transport": NETWORK_A["transport"]},
    {"source": "local-rtx5090", "destination": 50478207,
     "application_rtt_ms": NETWORK_B["median_rtt_s"] * 1000,
     "p95_rtt_ms": NETWORK_B["p95_rtt_s"] * 1000,
     "jitter_stddev_ms": NETWORK_B["jitter_stddev_s"] * 1000,
     "throughput_mbps": NETWORK_B["throughput"], "transport": NETWORK_B["transport"]},
    {"source": 50478198, "destination": 50478207,
     "application_rtt_ms": NETWORK_AB["network"]["median_rtt_s"] * 1000,
     "p95_rtt_ms": None,
     "jitter_stddev_ms": NETWORK_AB["network"]["jitter_stddev_s"] * 1000,
     "throughput_mbps": NETWORK_AB["network"]["throughput"],
     "transport": NETWORK_AB["network"]["transport"]},
]


def trace_for(run_dir: Path) -> dict | None:
    path = run_dir / "server.stderr.log"
    if not path.exists():
        return None
    result = summarize_trace(path)
    intervals = result["steady_single_token_intervals"]
    if intervals:
        cycles = [item["cycle_us"] for item in intervals]
        waits = [item["sum_response_wait_us_may_overlap"] for item in intervals]
        result["decomposition"] = {
            "median_cycle_ms": statistics.median(cycles) / 1000,
            "median_response_wait_sum_ms": statistics.median(waits) / 1000,
            "median_cycle_minus_wait_sum_ms": (statistics.median(cycles) - statistics.median(waits)) / 1000,
            "response_wait_fraction_of_median_cycle": statistics.median(waits) / statistics.median(cycles),
            "warning": "Response wait sums may overlap; residual includes local compute, serialization, and orchestration.",
        }
    else:
        result["decomposition"] = None
    return result


def router_trace_path(configuration: dict) -> Path | None:
    deployment = configuration.get("deployment") or {}
    text = deployment.get("router_command", "")
    match = re.search(r"/workspace/e026/(wan-router-services-[^/ ]+)-routing\.jsonl", text)
    if not match:
        return None
    path = ROOT / "remote-collected-final-002" / "a" / f"{match.group(1)}-routing.jsonl"
    return path if path.exists() else None


def router_summary(path: Path | None) -> dict | None:
    if path is None:
        return None
    rows = jsonl(path)
    workers = {}
    for worker in (0, 1):
        selected = [row for row in rows if row.get("worker") == worker]
        gets = [row["elapsed_s"] * 1000 for row in selected if row.get("cmd") == 8]
        workers[str(worker)] = {
            "get_tensor_operation_count": len(gets),
            "median_get_tensor_operation_ms_compute_plus_transport": statistics.median(gets) if gets else None,
            "request_protocol_bytes": sum(row.get("request_bytes", 0) for row in selected),
            "response_protocol_bytes": sum(row.get("response_bytes", 0) for row in selected),
        }
    peer = [row for row in rows if row.get("worker") == 1]
    return {
        "source": str(path),
        "workers": workers,
        "peer_wan_protocol_bytes": sum(row.get("request_bytes", 0) + row.get("response_bytes", 0) for row in peer),
    }


def gpu_stats(start: datetime, end: datetime) -> dict:
    result = {}
    for role in ("a", "b"):
        path = ROOT / "remote-collected-final-002" / role / "wan-services-001-gpu.csv"
        samples = []
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                if start <= stamp <= end:
                    samples.append({
                        "util": float(row[" utilization.gpu [%]"].split()[0]),
                        "memory": float(row[" memory.used [MiB]"].split()[0]),
                        "power": float(row[" power.draw [W]"].split()[0]),
                    })
        result[role] = {
            "samples": len(samples),
            "mean_gpu_utilization_percent": statistics.fmean(item["util"] for item in samples) if samples else None,
            "p95_gpu_utilization_percent": pct([item["util"] for item in samples], .95),
            "max_memory_used_mib": max((item["memory"] for item in samples), default=None),
            "mean_power_w": statistics.fmean(item["power"] for item in samples) if samples else None,
        }
    return result


def correctness(run_id: str, prompt_id: str, output: dict | None) -> dict:
    if output is None:
        return {"status": "NO_COMPLETE_OUTPUT", "token_stream_equal": False}
    if run_id == "local-target-001":
        return {"status": "TRUSTED_LOCAL_REFERENCE", "token_stream_equal": True}
    if run_id == "final-local-sealed-control-001":
        return {"status": "SEALED_LOCAL_REFERENCE", "token_stream_equal": True}
    reference_path = RUNS / "local-target-001" / prompt_id / "output.json"
    if not reference_path.exists():
        return {"status": "NO_MATCHED_REFERENCE", "token_stream_equal": None}
    reference = load(reference_path)["tokens"]
    observed = output["tokens"]
    mismatch = next((index for index, pair in enumerate(zip(reference, observed)) if pair[0] != pair[1]), None)
    equal = len(reference) >= len(observed) and mismatch is None
    return {"status": "PASS" if equal else "FAIL", "token_stream_equal": equal,
            "compared_tokens": min(len(reference), len(observed)), "first_mismatch": mismatch,
            "reference_run_id": "local-target-001"}


def estimated_cumulative(timestamp: str) -> float:
    when = dt(timestamp)
    ledger = jsonl(ROOT / "cost" / "ledger.jsonl")
    total = 0.0
    for lease_path in (ROOT / "cost" / "leases").glob("*.json"):
        lease = load(lease_path)
        attempts = [row for row in ledger if row["event"] == "CREATE_ATTEMPT" and row.get("label") == lease["label"]]
        if not attempts or dt(attempts[0]["timestamp"]) > when:
            continue
        start = dt(attempts[0]["timestamp"])
        stops = [row for row in ledger if row["event"] == "ABSENCE_CONFIRMED" and row.get("label") == lease["label"]]
        stop = min(when, dt(stops[-1]["timestamp"])) if stops else when
        total += max(0.0, (stop - start).total_seconds()) / 3600 * lease["hourly_usd"]
        total += lease["network_reserve_usd"]
    return total


def run_rows() -> list[dict]:
    rows = []
    for metrics_path in sorted(RUNS.glob("*/*/metrics.json")):
        metrics = load(metrics_path)
        run_dir = metrics_path.parent.parent
        prompt_dir = metrics_path.parent
        # Several legacy metrics records copied the prompt id into run_id.  The
        # immutable run-directory name is the authoritative execution id.
        run_id = run_dir.name
        configuration = load(run_dir / "configuration.json") if (run_dir / "configuration.json").exists() else {}
        output = load(prompt_dir / "output.json") if (prompt_dir / "output.json").exists() else None
        startup = load(run_dir / "startup.json") if (run_dir / "startup.json").exists() else {}
        trace = trace_for(run_dir)
        router = router_summary(router_trace_path(configuration))
        wan = run_id.startswith("wan-")
        timings = metrics.get("server_timings") or {}
        proposals = metrics.get("speculative_proposals")
        acceptances = metrics.get("speculative_acceptances")
        verifications = timings.get("draft_verification_steps") if proposals is not None else metrics["generated_tokens"]
        traversal_measurement = (
            "MEASURED_DRAFT_VERIFICATION_COUNTER" if proposals is not None and verifications is not None
            else "UNAVAILABLE_PRE_COUNTER_BUILD" if proposals is not None
            else "DERIVED_ONE_TARGET_EVALUATION_PER_COMMITTED_TOKEN"
        )
        client_sent = trace["total_sent_protocol_bytes"] if trace else None
        client_received = trace["total_received_protocol_bytes"] if trace else None
        median_steady = None
        if trace and trace["median_single_token_sent_bytes"] is not None:
            median_steady = trace["median_single_token_sent_bytes"] + trace["median_single_token_received_bytes"]
        if not wan:
            bytes_per_token = None
            byte_scope = "none"
            bytes_measurement = "NOT_APPLICABLE_LOCAL"
        elif median_steady is not None:
            bytes_per_token = median_steady
            byte_scope = "median steady coordinator-to-first-remote RPC protocol bytes"
            bytes_measurement = "MEASURED_MEDIAN_STEADY_RPC_PROTOCOL"
        elif trace:
            bytes_per_token = (client_sent + client_received) / metrics["generated_tokens"]
            byte_scope = "coordinator RPC protocol total including startup divided by committed tokens"
            bytes_measurement = "MEASURED_TOTAL_INCLUDING_STARTUP_RPC_PROTOCOL"
        else:
            bytes_per_token = None
            byte_scope = "coordinator-to-first-remote RPC protocol"
            bytes_measurement = "UNAVAILABLE_NO_PROTOCOL_TRACE"
        timestamp = metrics["timestamp"]
        duration = metrics.get("total_s") or 0.0
        start = dt(timestamp).astimezone(timezone.utc) - __import__("datetime").timedelta(seconds=duration)
        row = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "source_recorded_run_id": metrics.get("run_id"),
            "prompt_id": metrics["prompt_id"],
            "timestamp": timestamp,
            "evidence_class": metrics.get("evidence_class"),
            "repository_commit": REPOSITORY_COMMIT,
            "model": MODEL,
            "llama_cpp": LLAMA,
            "configuration_hash": metrics.get("configuration_hash"),
            "exact_command": configuration.get("command"),
            "configuration": metrics.get("configuration"),
            "node_identifiers": [node["node_id"] for node in NODES] if wan else ["local-rtx5090"],
            "gpu_types": [node["gpu"] for node in NODES] if wan else [NODES[0]["gpu"]],
            "regions": [node["region"] for node in NODES] if wan else [NODES[0]["region"]],
            "stage_assignments": STAGES if wan else [{"node_id": "local-rtx5090", "layers": [0, 63]}],
            "pairwise_network": PAIRWISE if wan else [],
            "context_length": metrics["context_tokens"],
            "generated_tokens": metrics["generated_tokens"],
            "ttft_s": metrics["ttft_s"],
            "prompt_tok_s": metrics.get("prompt_tok_s"),
            "decode_tok_s": metrics["decode_tok_s"],
            "median_tpot_s": metrics["median_tpot_s"],
            "p95_tpot_s": metrics["p95_tpot_s"],
            "bytes_sent": client_sent,
            "bytes_received": client_received,
            "bytes_per_committed_token": bytes_per_token,
            "byte_scope": byte_scope,
            "bytes_measurement": bytes_measurement,
            "target_traversals": verifications,
            "target_traversals_measurement": traversal_measurement,
            "wan_traversals": verifications * 2 if wan else 0,
            "wan_traversal_definition": "two physical inter-machine stage boundaries per full target evaluation" if wan else None,
            "speculative_proposals": proposals,
            "speculative_acceptances": acceptances,
            "acceptance_rate": acceptances / proposals if proposals else None,
            "accepted_tokens_per_verification": acceptances / verifications if acceptances is not None and verifications else None,
            "tokens_per_expensive_target_traversal": metrics["generated_tokens"] / verifications if verifications else None,
            "stage_compute_times": None,
            "stage_network_waits": trace.get("decomposition") if trace else None,
            "pipeline_idle": {"inferred_from_response_wait_fraction": trace.get("decomposition", {}).get("response_wait_fraction_of_median_cycle")}
                if trace and trace.get("decomposition") else None,
            "gpu_utilization": gpu_stats(start, dt(timestamp)) if wan else None,
            "router_operations": router,
            "state_size_bytes": STATE["save"]["state_bytes"],
            "checkpoint_time_s": STATE["save"]["serialization_s"],
            "restore_time_s": STATE["restore"]["restore_s"],
            "failure_timing": RECOVERY if run_id == "wan-replica-kill-001" else None,
            "correctness": correctness(run_id, metrics["prompt_id"], output),
            "output_token_hash": metrics["output_token_hash"],
            "startup": startup,
            "cost": {"estimated_cumulative_upper_usd": estimated_cumulative(timestamp),
                     "allocation": "continuous experiment-level spend; not attributed to one request"} if wan else {"usd": 0.0},
            "source_metrics": str(metrics_path),
        }
        rows.append(row)
    return rows


def ready_event(path: Path) -> dict:
    return next(row for row in reversed(jsonl(path)) if row.get("event") == "SHARD_READY")


def cold_start() -> dict:
    ledger = jsonl(ROOT / "cost" / "ledger.jsonl")
    result = {}
    for role, instance_id, label in (("a", 50478198, "e026-a-1789033163"),
                                     ("b", 50478207, "e026-b-1789033167")):
        attempts = next(row for row in ledger if row["event"] == "CREATE_ATTEMPT" and row.get("label") == label)
        ssh_rows = jsonl(ROOT / "remote" / str(instance_id) / "ssh.jsonl")
        first = next(row for row in ssh_rows if row["returncode"] == 0)
        built = next(row for row in ssh_rows if row["returncode"] == 0 and row["command"].startswith("test -x"))
        stage = next(row for row in ssh_rows if row["returncode"] == 0 and "ggml-rpc-server -H" in row["command"])
        start = dt(attempts["timestamp"])
        result[role] = {
            "instance_id": instance_id,
            "create_attempt_timestamp": attempts["timestamp"],
            "first_successful_ssh_timestamp": first["timestamp"],
            "provision_to_ssh_s": (dt(first["timestamp"]) - start).total_seconds(),
            "build_verified_timestamp": built["timestamp"],
            "provision_to_build_verified_s": (dt(built["timestamp"]) - start).total_seconds(),
            "stage_launch_timestamp": stage["timestamp"],
            "provision_to_stage_launch_s": (dt(stage["timestamp"]) - start).total_seconds(),
        }
    shard_a = ready_event(ROOT / "remote-collected-final-002" / "a" / "shard-a-cold-001.jsonl")
    shard_b = ready_event(ROOT / "remote-collected-final-002" / "b" / "shard-b-cold-001.jsonl")
    upgrade_a = ready_event(ROOT / "remote-collected-final-002" / "a" / "shard-a-all-upgrade-002.jsonl")
    upgrade_b = ready_event(ROOT / "remote-collected-final-002" / "b" / "shard-b-all-upgrade-002.jsonl")
    standby_cold = ready_event(ROOT / "remote-collected-final-002" / "b" / "standby-a-cache-001.jsonl")
    standby_warm = ready_event(ROOT / "remote-collected-final-002" / "b" / "standby-a-cache-verify-002.jsonl")
    direct_warm = load(RUNS / "wan-direct-cache-pipeline-no-checkpoints-001" / "startup.json")
    sealed_warm = load(RUNS / "final-wan-sealed-001" / "startup.json")
    deployed = dt(DEPLOYMENT["timestamp"])
    relay_ready = dt(load(ROOT / "services" / "wan-router-services-final-001" / "readiness.json")["timestamp"])
    result.update({
        "cold_shard_acquisition": {"a": shard_a, "b": shard_b, "exact_cache_upgrade_a": upgrade_a,
                                   "exact_cache_upgrade_b": upgrade_b},
        "standby_same_shard_cold": standby_cold,
        "standby_same_shard_disk_warm_verify": standby_warm,
        "disk_warm_same_shard_speedup": standby_cold["elapsed_s"] / standby_warm["elapsed_s"],
        "best_disk_warm_full_server_ready_s": direct_warm["ready_seconds"],
        "sealed_disk_warm_full_server_ready_s": sealed_warm["ready_seconds"],
        "gpu_warm_relay_registration_s": (relay_ready - deployed).total_seconds(),
        "full_cold_topology_critical_node_s": max(result["a"]["provision_to_stage_launch_s"],
                                                   result["b"]["provision_to_stage_launch_s"]),
    })
    result["full_cold_to_best_disk_warm_speedup"] = (
        result["full_cold_topology_critical_node_s"] / result["best_disk_warm_full_server_ready_s"])
    return result


def final_stage_compute() -> dict:
    values = {}
    paths = {
        "worker_a": (ROOT / "remote-collected-final-002" / "a" / "wan-stage-a-restart-after-kill-001.log", None),
        "worker_b": (ROOT / "remote-collected-final-002" / "b" / "wan-services-001-stage.log", 285),
    }
    for name, (path, tail) in paths.items():
        timings = [(int(end) - int(begin)) / 1000 for begin, end in re.findall(
            r"E026_STAGE .*?begin_us=(\d+) end_us=(\d+)", path.read_text(encoding="utf-8", errors="replace"))]
        if tail:
            timings = timings[-tail:]
        values[name] = {"samples": len(timings), "median_ms": statistics.median(timings),
                        "mean_ms": statistics.fmean(timings), "p95_ms": pct(timings, .95), "max_ms": max(timings)}
    return values


def final_row() -> dict:
    trace = FINAL_PARTIAL["trace"]
    router = router_summary(ROOT / "remote-collected-final-002" / "a" / "wan-router-services-final-001-routing.jsonl")
    coordinator = trace["total_sent_protocol_bytes"] + trace["total_received_protocol_bytes"]
    total_wan = coordinator + router["peer_wan_protocol_bytes"]
    steady_coordinator = trace["median_single_token_sent_bytes"] + trace["median_single_token_received_bytes"]
    peer_rows = jsonl(ROOT / "remote-collected-final-002" / "a" / "wan-router-services-final-001-routing.jsonl")
    starts = [index for index, row in enumerate(peer_rows) if row.get("worker") == 1 and row.get("cmd") == 16]
    cycles = []
    for left, right in zip(starts, starts[1:]):
        selected = [row for row in peer_rows[left:right] if row.get("worker") == 1]
        cycles.append(sum(row.get("request_bytes", 0) + row.get("response_bytes", 0) for row in selected))
    failure_time = dt(FINAL_PARTIAL["failure"]["timestamp"])
    start_time = failure_time - __import__("datetime").timedelta(seconds=FINAL_PARTIAL["last_token_arrival_s"] + 36.4)
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": "final-wan-sealed-001",
        "source_recorded_run_id": None,
        "prompt_id": "sealed-01-factual",
        "timestamp": FINAL_PARTIAL["failure"]["timestamp"],
        "evidence_class": "PHYSICAL",
        "repository_commit": REPOSITORY_COMMIT,
        "model": MODEL,
        "llama_cpp": LLAMA,
        "configuration_hash": SEAL["configuration_sha256"],
        "exact_command": load(RUNS / "final-wan-sealed-001" / "configuration.json")["command"],
        "configuration": SEAL["inference"],
        "node_identifiers": [node["node_id"] for node in NODES],
        "gpu_types": [node["gpu"] for node in NODES],
        "regions": [node["region"] for node in NODES],
        "stage_assignments": STAGES,
        "pairwise_network": PAIRWISE,
        "context_length": 56,
        "requested_generated_tokens": 512,
        "generated_tokens": FINAL_PARTIAL["completed_tokens"],
        "terminal_event_observed": False,
        "ttft_s": FINAL_PARTIAL["ttft_s"],
        "prompt_tok_s": 2.4288935977617747,
        "decode_tok_s": FINAL_PARTIAL["partial_decode_tok_s"],
        "median_tpot_s": FINAL_PARTIAL["median_tpot_s"],
        "p95_tpot_s": FINAL_PARTIAL["p95_tpot_s"],
        "bytes_sent": trace["total_sent_protocol_bytes"],
        "bytes_received": trace["total_received_protocol_bytes"],
        "coordinator_wan_protocol_bytes": coordinator,
        "peer_wan_protocol_bytes": router["peer_wan_protocol_bytes"],
        "total_wan_protocol_bytes": total_wan,
        "bytes_per_committed_token_including_startup_all_wan_links": total_wan / FINAL_PARTIAL["completed_tokens"],
        "median_steady_coordinator_bytes_per_token": steady_coordinator,
        "median_steady_peer_bytes_per_token": statistics.median(cycles),
        "median_steady_all_wan_links_bytes_per_token": steady_coordinator + statistics.median(cycles),
        "bytes_per_committed_token": steady_coordinator + statistics.median(cycles),
        "byte_scope": "median steady RPC protocol bytes across both physical WAN stage boundaries",
        "bytes_measurement": "MEASURED_RPC_PROTOCOL",
        "byte_definition": "RPC protocol bytes excluding TCP/IP/SSH framing; all-WAN total sums local-to-A and A-to-B",
        "target_traversals": FINAL_PARTIAL["completed_tokens"],
        "target_traversals_measurement": "DERIVED_ONE_TARGET_EVALUATION_PER_COMMITTED_TOKEN",
        "wan_traversals": FINAL_PARTIAL["completed_tokens"] * 2,
        "wan_traversal_definition": "two physical inter-machine stage boundaries per full target evaluation",
        "speculative_proposals": 0,
        "speculative_acceptances": 0,
        "acceptance_rate": None,
        "accepted_tokens_per_verification": None,
        "tokens_per_expensive_target_traversal": 1.0,
        "stage_compute_times": final_stage_compute(),
        "stage_network_waits": trace.get("decomposition") or trace_for(RUNS / "final-wan-sealed-001")["decomposition"],
        "pipeline_idle": {"response_wait_fraction_of_median_cycle": trace_for(RUNS / "final-wan-sealed-001")["decomposition"]["response_wait_fraction_of_median_cycle"]},
        "gpu_utilization": gpu_stats(start_time, failure_time),
        "router_operations": router,
        "state_size_bytes": STATE["save"]["state_bytes"],
        "checkpoint_time_s": STATE["save"]["serialization_s"],
        "restore_time_s": STATE["restore"]["restore_s"],
        "failure_timing": FINAL_PARTIAL["failure"],
        "correctness": {"status": "FAIL_STRICT_GREEDY_AND_INCOMPLETE", "matched_prefix_tokens": 178,
                        "first_mismatch": FINAL_PARTIAL["first_mismatch"],
                        "identical_history_logits": FINAL_PARTIAL["identical_history_logits"]},
        "output_token_hash": FINAL_PARTIAL["partial_output_token_hash"],
        "startup": load(RUNS / "final-wan-sealed-001" / "startup.json"),
        "cost": {"actual_total_experiment_usd": OPENING["budget"]["conservative_available_usd"] - FINAL_PROVIDER["budget"]["conservative_available_usd"],
                 "estimated_cumulative_upper_usd": estimated_cumulative(FINAL_PARTIAL["failure"]["timestamp"])},
        "classification": FINAL_PARTIAL["classification"],
        "source_analysis": str(ROOT / "analysis" / "final-wan-sealed-001-partial-logits-trace.json"),
    }


def main() -> None:
    rows = run_rows()
    rows.append(final_row())
    metrics_path = ROOT / "metrics-canonical-v3.jsonl"
    write_jsonl_once(metrics_path, rows)

    actual_cost = OPENING["budget"]["conservative_available_usd"] - FINAL_PROVIDER["budget"]["conservative_available_usd"]
    conservative_cost = estimated_cumulative(FINAL_PROVIDER["timestamp"])
    startup = cold_start()
    stage_compute = final_stage_compute()
    ordinary = next(row for row in rows if row["run_id"] == "wan-ordinary-profile-001")
    direct = next(row for row in rows if row["run_id"] == "wan-direct-cache-pipeline-no-checkpoints-001")
    direct_mtp = next(row for row in rows if row["run_id"] == "wan-direct-cache-pipeline-mtp-k3-001")
    proxy_mtp = next(row for row in rows if row["run_id"] == "wan-routed-cache-pipeline-mtp-k3-001")
    final = rows[-1]
    recovery_resumed_at = dt(RECOVERY["failure_injection"]["timestamp"]) + timedelta(
        seconds=RECOVERY["user_visible_interruption_s"])
    summary = {
        "canonical_verdict": VERDICT,
        "experiment_id": EXPERIMENT_ID,
        "major_results": {
            "local_sealed_512_tok_s": next(row["decode_tok_s"] for row in rows if row["run_id"] == "final-local-sealed-control-001"),
            "ordinary_wan_tok_s": ordinary["decode_tok_s"],
            "best_target_only_wan_tok_s": direct["decode_tok_s"],
            "best_exact_wan_tok_s": direct["decode_tok_s"],
            "best_attempted_inexact_mtp_wan_tok_s": direct_mtp["decode_tok_s"],
            "sealed_partial_wan_tok_s": final["decode_tok_s"],
            "sealed_completed_of_requested_tokens": [final["generated_tokens"], final["requested_generated_tokens"]],
            "target_only_speedup_over_ordinary": direct["decode_tok_s"] / ordinary["decode_tok_s"],
            "best_attempted_inexact_mtp_speedup_over_ordinary": direct_mtp["decode_tok_s"] / ordinary["decode_tok_s"],
            "stable_proxy_mtp_speedup": proxy_mtp["decode_tok_s"] / next(row["decode_tok_s"] for row in rows if row["run_id"] == "wan-routed-cache-pipeline-no-checkpoints-001"),
            "best_tokens_per_target_traversal": max(row["tokens_per_expensive_target_traversal"] or 0 for row in rows if row["run_id"].startswith("wan-")),
            "final_first_greedy_mismatch": final["correctness"]["first_mismatch"],
            "recovery_interruption_s": RECOVERY["user_visible_interruption_s"],
            "disk_warm_same_shard_speedup": startup["disk_warm_same_shard_speedup"],
            "actual_vast_cost_usd": actual_cost,
            "conservative_upper_cost_usd": conservative_cost,
        },
        "startup": startup,
        "state": STATE,
        "recovery": RECOVERY,
        "stage_compute": stage_compute,
        "gates": {
            "genuine_wan": {"pass": True, "machines": 3, "max_measured_application_rtt_ms": max(x["application_rtt_ms"] for x in PAIRWISE)},
            "distributed_correctness": {"pass": False, "reason": "sealed greedy stream first differed at token index 178"},
            "exact_speculative_correctness": {"pass": False, "reason": "native MTP diverged on 4 of 6 development prompts and was rejected"},
            "synchronization_compression": {"development_pass": True, "sealed_pass": False,
                                             "development_best_tokens_per_traversal": max(direct_mtp["tokens_per_expensive_target_traversal"], proxy_mtp["tokens_per_expensive_target_traversal"]),
                                             "sealed_tokens_per_traversal": 1.0},
            "interactive_decode": {"pass": False, "required_tok_s": 8.0, "best_measured_tok_s": direct_mtp["decode_tok_s"],
                                   "best_exact_tok_s": direct["decode_tok_s"],
                                   "best_attempted_inexact_mtp_tok_s": direct_mtp["decode_tok_s"],
                                   "sealed_partial_tok_s": final["decode_tok_s"]},
            "ordinary_baseline_improvement": {"development_pass": direct["decode_tok_s"] / ordinary["decode_tok_s"] >= 1.5,
                                               "sealed_pass": final["decode_tok_s"] / ordinary["decode_tok_s"] >= 1.5},
            "cached_startup": {"pass": startup["disk_warm_same_shard_speedup"] >= 3,
                               "same_shard_speedup": startup["disk_warm_same_shard_speedup"]},
            "failure_survival": {"pass": RECOVERY["status"] == "PASS" and RECOVERY["user_visible_interruption_s"] < 20,
                                 "interruption_s": RECOVERY["user_visible_interruption_s"]},
            "sealed_512_tokens": {"pass": False, "completed": final["generated_tokens"], "requested": 512},
            "budget": {"pass": actual_cost <= 38, "actual_usd": actual_cost, "ceiling_usd": 38},
        },
        "next_decision": "continue solving a specific primitive",
        "blocking_primitive": "an exact, reconnectable, state-local multi-token verification protocol that removes per-token RPC round trips and survives transport reconnection",
    }
    summary["synthesis_lineage"] = {
        "canonical_dataset": str(metrics_path),
        "supersedes": ["metrics.jsonl", "metrics-canonical.jsonl", "metrics-canonical-v2.jsonl",
                       "summary.json", "summary-canonical.json", "summary-final.json",
                       "final-receipt.json", "final-receipt-canonical.json",
                       "final-receipt-canonical-v2.json"],
        "reason": "canonical run-directory IDs, consistent byte denominators with explicit scope, complete local host inventory, correct recovery timestamp semantics, and exact-vs-inexact throughput labels",
    }
    write_once(ROOT / "summary-canonical-v3.json", summary)

    dirty = SEAL["dirty_tree"]
    receipt = {
        "canonical_verdict": VERDICT,
        "experiment_id": EXPERIMENT_ID,
        "repository_commit": REPOSITORY_COMMIT,
        "dirty_tree_state": "DIRTY",
        "dirty_tree_porcelain_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "dirty_tree_snapshot_source": "seal/configuration.json",
        "model": MODEL,
        "llama_cpp": LLAMA,
        "configuration_hash": SEAL["configuration_sha256"],
        "exact_final_command": final["exact_command"],
        "exact_final_inference_config": SEAL["inference"],
        "nodes": NODES,
        "stage_assignments": STAGES,
        "pairwise_network": PAIRWISE,
        "final_prompt": {"id": SEAL["final"]["prompt_id"], "content_sha256": SEAL["final"]["prompt_content_sha256"]},
        "generated_tokens": {"requested": 512, "completed": final["generated_tokens"], "terminal_event": False},
        "output_token_hash": {"partial_282": final["output_token_hash"],
                              "local_sealed_512": next(row["output_token_hash"] for row in rows if row["run_id"] == "final-local-sealed-control-001")},
        "ttft_s": final["ttft_s"],
        "median_tpot_s": final["median_tpot_s"],
        "p95_tpot_s": final["p95_tpot_s"],
        "committed_tok_s_partial": final["decode_tok_s"],
        "speculation": {"method": "none", "proposals": 0, "acceptances": 0},
        "tokens_per_expensive_target_traversal": 1.0,
        "bytes": {key: final[key] for key in ("bytes_sent", "bytes_received", "coordinator_wan_protocol_bytes",
                                               "peer_wan_protocol_bytes", "total_wan_protocol_bytes",
                                               "bytes_per_committed_token_including_startup_all_wan_links",
                                               "median_steady_all_wan_links_bytes_per_token", "byte_definition")},
        "stage_compute_times": stage_compute,
        "gpu_utilization": final["gpu_utilization"],
        "startup": startup,
        "hybrid_model_state": {
            "total_bytes": STATE["save"]["state_bytes"],
            "attention_state_bytes": STATE["save"]["attention_state_bytes"],
            "recurrent_state_with_header_bytes": STATE["save"]["recurrent_state_with_header_bytes"],
            "checkpoint_serialization_s": STATE["save"]["serialization_s"],
            "restore_gpu_s": STATE["restore"]["restore_s"],
        },
        "failure_injection": {"timestamp": RECOVERY["failure_injection"]["timestamp"],
                              "node_id": RECOVERY["failure_injection"]["node_id"],
                              "committed_token": RECOVERY["failure_injection"]["committed_tokens_observed"]},
        "recovery_completion": {"timestamp": recovery_resumed_at.isoformat(),
                                "timestamp_basis": "failure injection wall time plus measured token-stream interruption",
                                "recovery_receipt_generated_at": RECOVERY["timestamp"],
                                "user_visible_interruption_s": RECOVERY["user_visible_interruption_s"],
                                "duplicates": RECOVERY["duplicate_tokens"], "lost": RECOVERY["lost_tokens"],
                                "full_prompt_replay_tokens": RECOVERY["full_prompt_replay_tokens"],
                                "result": RECOVERY["status"]},
        "final_correctness": final["correctness"],
        "final_failure": FINAL_PARTIAL["failure"],
        "cost": {"actual_vast_expenditure_usd": actual_cost,
                 "conservative_upper_estimate_usd": conservative_cost,
                 "hard_ceiling_usd": 38.0,
                 "final_available_credit_usd": FINAL_PROVIDER["budget"]["conservative_available_usd"],
                 "all_e026_instances_absent": FINAL_PROVIDER["all_e026_instances_absent"]},
        "gates": summary["gates"],
        "dataset": {"path": str(metrics_path), "sha256": file_digest(metrics_path), "rows": len(rows)},
        "remote_evidence_collection": {"receipt": "remote-collected-final-002/receipt.json",
                                       "files": len(COLLECTION["files"]), "bytes": COLLECTION["total_bytes"]},
        "synthesis_lineage": summary["synthesis_lineage"],
        "local_host_inventory_source": "resource/local-host-post-run-001.json",
        "canonicality": {"status": "CANONICAL", "dataset": str(metrics_path)},
        "timestamp": utc_now(),
    }
    write_once(ROOT / "final-receipt-canonical-v3.json", receipt)
    print(json.dumps({"verdict": VERDICT, "metric_rows": len(rows), "actual_cost_usd": actual_cost,
                      "conservative_upper_usd": conservative_cost, "dataset_sha256": receipt["dataset"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
