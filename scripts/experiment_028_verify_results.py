"""Independent arithmetic and artifact checks for completed E028 evidence."""
import json
import math
from collections import Counter, defaultdict

from swarm_inference.experiments.experiment_028.local import OUT, write_json


def main():
    def read(name):
        return json.loads((OUT/name).read_text(encoding="utf8"))

    config, summary = read("config.json"), read("summary.json")
    correctness, stress = read("correctness_results.json"), read("rollback_stress_results.json")
    assert stress["complete"] and len(stress["scenarios"]) == 9
    assert sum(r["forced_events"] for r in stress["scenarios"]) == 360
    fingerprint_pairs=0
    for scenario in stress["scenarios"]:
        raw=read(scenario["trace"])
        for restore in raw["rollback_checks"]:
            assert restore["restored_position"] == restore["boundary_position"] + restore["accepted_replay_rows"]
            assert all(restore["replay_output_checks"])
            if restore["before_hashes"] is not None:
                assert restore["before_hashes"] == restore["after_hashes"]
                fingerprint_pairs+=len(restore["before_hashes"])
    rows = [json.loads(line) for line in (OUT/"wan_sweep_results.jsonl").read_text().splitlines()]
    aggregates = read("wan_aggregate.json")
    assert Counter(p["category"] for p in read("prompts.json")) == dict(conversational=2, coding=2, reasoning=2, **{"long-context":2})
    assert correctness["complete"] and len(correctness["runs"]) == 128
    assert all(r["committed_tokens"] == 256 for r in correctness["runs"])
    assert len(rows) == 1920 and len(aggregates) == 80
    assert len({(r["network"],r["prompt_id"],r["k"],r["w"],r["seed"]) for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        assert row["committed_tokens"] == 256
        assert row["evidence_class"] == "PHYSICALLY GROUNDED MODEL"
        assert row["physical_gpu_count"] == 1 and row["virtual_target_gpu_count"] == 3
        assert 1 <= row["peak_inflight_chunks"] <= row["w"]
        for metric in ["wan_wait_fraction","stage_a_utilization","stage_b_utilization","stage_c_utilization",
                       "speculative_acceptance_rate","discarded_speculative_compute_fraction","scheduler_overhead_fraction"]:
            assert math.isfinite(row[metric]) and -1e-9 <= row[metric] <= 1+1e-9, (metric,row[metric])
        assert math.isclose(row["committed_tokens_per_second"],256000/row["elapsed_ms"],rel_tol=1e-12)
        groups[(row["network"],row["k"],row["w"])].append(row)
    for agg in aggregates:
        group=groups[(agg["network"],agg["k"],agg["w"])]
        assert len(group)==24
        assert len({r["prompt_id"] for r in group})==8
        assert {r["seed"] for r in group}==set(config["network_seeds"])
        tokens=sum(r["committed_tokens"] for r in group)
        wall=sum(r["elapsed_ms"] for r in group)
        assert math.isclose(agg["committed_tokens_per_second"],tokens*1000/wall,rel_tol=1e-12)
        discard=sum(r["discarded_target_compute_ms"] for r in group)/sum(r["target_compute_ms"] for r in group)
        assert math.isclose(agg["discarded_speculative_compute_fraction"],discard,rel_tol=1e-12,abs_tol=1e-12)
    best=max((r for r in aggregates if r["network"]=="WAN-60" and r["w"]>1),key=lambda r:r["committed_tokens_per_second"])
    sync=max(r["committed_tokens_per_second"] for r in aggregates if r["network"]=="WAN-60" and r["w"]==1 and r["k"])
    zero=max(r["committed_tokens_per_second"] for r in aggregates if r["network"]=="LOCAL" and r["k"])
    assert (summary["best_k"],summary["best_w"])==(best["k"],best["w"])
    assert math.isclose(summary["async_speedup_60ms"],best["committed_tokens_per_second"]/sync,rel_tol=1e-12)
    assert math.isclose(summary["throughput_retention_60ms"],best["committed_tokens_per_second"]/zero,rel_tol=1e-12)
    for case in read("simulator_validation.json")["conditions"]:
        error=abs(case["predicted_runtime_ms"]-case["measured_runtime_ms"])/case["measured_runtime_ms"]
        assert math.isclose(error,case["absolute_relative_error"],rel_tol=1e-12)
        assert case["passed"] == (error<=.10)
    intervals=0
    for path in (OUT/"traces").glob("virtual-wan60-*.json"):
        trace=json.loads(path.read_text())
        for resource in ["A","B","C","draft","cpu"]:
            ops=[o for o in trace["operations"] if o["resource"]==resource]
            for left,right in zip(ops,ops[1:]):
                assert left["end_ms"] <= right["start_ms"]+1e-8, (path,resource,left,right)
                intervals+=1
    gates=all(summary[k] for k in ["correctness_passed","rollback_passed","genericity_passed","simulator_validated"])
    expected="PASS_STRONG" if gates and all(summary["pass_strong_criteria"].values()) else "PASS" if gates and all(summary["pass_criteria"].values()) else "FAIL"
    assert summary["verdict"]==expected
    required=["README.md","config.json","prompts.json","environment.json","stage_profile.json","correctness_results.json",
              "rollback_stress_results.json","simulator_validation.json","wan_sweep_results.jsonl","summary.json","report.md"]
    assert all((OUT/name).is_file() for name in required)
    assert len(list((OUT/"plots").glob("*.png")))==6
    result=dict(passed=True,physical_runs=128,physical_committed_tokens=32768,virtual_runs=len(rows),aggregate_groups=len(aggregates),
                resource_nonoverlap_pairs_checked=intervals,stage_full_and_recurrent_fingerprint_pairs=fingerprint_pairs,forced_events=sum(r["forced_events"] for r in stress["scenarios"]),
                note="Independent arithmetic/artifact audit; experiment success is separately determined by summary.json",verdict=summary["verdict"])
    write_json(OUT/"result_verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
