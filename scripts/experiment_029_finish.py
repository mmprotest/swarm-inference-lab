"""Finish artifacts after the physical workload, without starting another run."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/E029_LOCAL_COMMIT_ANCHORED_WAN_TREE'
while True:
    p=OUT/'progress.json'
    if p.exists() and json.loads(p.read_text()).get('completed_runs')==216 and (OUT/'drafter_profile.json').exists():break
    log=OUT/'run_final_console.txt'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text(errors='replace')[-4000:]:raise RuntimeError('Physical workload stopped; inspect run_final_console.txt')
    time.sleep(5)
for script in ('experiment_029_analyze.py','experiment_029_report.py'):
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
required=['README.md','config.json','environment.json','prompts.json','e028_baseline_import.json','drafter_profile.json','branch_state_correctness.json','tree_verification_profile.json','simulator_validation.json','tree_rounds.jsonl','wan_sweep_results.jsonl','oracle_results.json','summary.json','report.md']
assert all((OUT/p).is_file() and (OUT/p).stat().st_size for p in required)
assert len(list((OUT/'plots').glob('*.png')))==7
summary=json.loads((OUT/'summary.json').read_text())
assert summary['physical_runs']==216 and summary['physical_committed_tokens']==55296
with (OUT/'wan_sweep_results.jsonl').open() as f:assert sum(1 for _ in f)==1800
assert summary['verdict'] in ('PASS_STRONG','PASS','FAIL','BLOCKED')
print('E029 ARTIFACTS COMPLETE',summary['verdict'],summary['decision'],flush=True)
