"""Focused E028 evidence audit; does not execute any model or prior experiment."""
import hashlib
import json
import re
from pathlib import Path

from swarm_inference.experiments.experiment_028.local import ROOT, OUT, write_json


def main():
    runtime=list((ROOT/"src/swarm_inference/experiments/experiment_028").glob("*.py"))
    runtime.append(ROOT/"native/experiment_028/e028_stage_server.cpp")
    violations=[]
    for p in runtime:
        if p.name in {"report.py","collect.py","validation.py"}:
            continue
        for number,line in enumerate(p.read_text().splitlines(),1):
            if re.search(r"\b(if|elif|case)\b.*qwen",line,re.I):
                violations.append(dict(file=str(p.relative_to(ROOT)),line=number,text=line.strip()))
    genericity=dict(passed=not violations,violations=violations,
        inspected_sources=[str(p.relative_to(ROOT)) for p in runtime],
        backend_boundary="Existing architecture-specific graph builder and stage-range support inside pinned llama.cpp; E028 adds no model kernels.",
        state_api="llama_memory_seq_cp/rm; KV prefix sharing and recurrent copy-on-write; per-chunk cached activation input",
        position_api="llama_model_rope_type selects backend-defined text position layout",
        draft_api="Generic token + conditioning-hidden MTP interface; no model-name dispatch",
        limitation="One architecture/backend was exercised; this does not establish support for every GGUF architecture.")
    write_json(OUT/"genericity_audit.json",genericity)
    if not (OUT/"correctness_results.json").exists():
        return
    results=json.loads((OUT/"correctness_results.json").read_text())
    evidence=[]
    for row in results["runs"]:
        path=OUT/row["trace"]
        run=json.loads(path.read_text())
        generated=[]
        committed_position=len(run["prompt_tokens"])
        valid=True
        for chunk in run["chunks"]:
            if chunk["accepted"] is None:
                continue
            valid &= not chunk["invalidated"] and chunk["position"]==committed_position
            generated.extend(chunk["tokens"][:chunk["accepted"]])
            committed_position+=chunk["accepted"]
        valid &= generated==run["committed_tokens"]
        outstanding=max(sum(c["epoch"]==at["epoch"] and c["launched_ms"]<=at["launched_ms"] and
                            (c["completed_ms"] is None or c["completed_ms"]>at["launched_ms"]) for c in run["chunks"])
                        for at in run["chunks"])
        evidence.append(dict(prompt_id=row["prompt_id"],k=row["k"],w=row["w"],
                             causal_commit_audit_passed=bool(valid),peak_unfinished_target_verifications=outstanding,
                             trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(OUT/"causal_audit.json",dict(complete=results["complete"],passed=all(r["causal_commit_audit_passed"] for r in evidence),runs=evidence))
    sources=runtime+[ROOT/"scripts/experiment_028_run.py",ROOT/"scripts/experiment_028_prepare.py",
                     ROOT/"scripts/experiment_028_native_v2.py",ROOT/"scripts/experiment_028_build.cmd",
                     ROOT/"tests/experiments/test_experiment_028_simulator.py",ROOT/"native/experiment_028/CMakeLists.txt",
                     ROOT/"scripts/experiment_028_audit.py",ROOT/"scripts/experiment_028_finish_run.py",
                     ROOT/"scripts/experiment_028_verify_results.py",ROOT/"scripts/experiment_028_calibrate_idle.py",OUT/"config.json",OUT/"prompts.json",OUT/"methods.md",OUT/"README.md"]
    write_json(OUT/"provenance/final_sources.json",[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
    print(json.dumps(dict(genericity_passed=genericity["passed"],causal_runs=len(evidence),causal_passed=all(r["causal_commit_audit_passed"] for r in evidence))),flush=True)


if __name__=="__main__":
    main()
