# E029_LOCAL_COMMIT_ANCHORED_WAN_TREE

Read `report.md` for the decision and `summary.json` for machine-readable gates. WAN throughput is a physically grounded trace-driven model, not a real multi-GPU benchmark. E028 artifacts are imported and sealed by hash in `e028_baseline_import.json`.

Reproduce the analysis from the saved traces without model execution:

```powershell
python scripts/experiment_029_analyze.py
python scripts/experiment_029_report.py
python scripts/experiment_029_audit.py
```

For a new physical collection on this local installation, preserve this sealed output folder first and start with a fresh experiment output directory. Existing successful run files are reused by the runner, so do not mix native/runtime versions within one output directory. Then run from the repository root:

```powershell
python scripts/experiment_029_prepare.py
python scripts/experiment_029_build_native.py
cmd /c scripts\experiment_029_build.cmd
python scripts/experiment_029_profile.py
python scripts/experiment_029_validate.py
python scripts/experiment_029_run.py
python scripts/experiment_029_analyze.py
python scripts/experiment_029_report.py
```

The free 1.1 GB DFlash2 GGUF download is documented and hashed in `drafter_profile.json`; the existing target is reused. All execution is local. The native worker links the pinned existing llama.cpp build. `environment.json` records code/library versions and hashes. No credentials or paid resources are required.

`runs/` contains physical traces; `tree_rounds.jsonl` consolidates their round evidence. `wan_sweep_results.jsonl` contains per-prompt/per-seed projections, and `wan_aggregate.json` contains mechanically aggregated values. The eight versioned prompts are unchanged from E028. `compatibility_attempts/` and diagnostic JSON files retain rejected implementation attempts; they are excluded from final timing inputs. The final runtime uses one committed root, a canonical working KV sequence, partial-state checkpoints per active depth, and a measured local accepted-path replay where branching requires it. Generic linear chains use position checkpoints to avoid replay.

The full N/D cost grid and fixed decision operating points are specified in `config.json`. Local wall-clock throughput is never presented as three-GPU throughput. The seven figures in `plots/` are standalone scientific plots derived from the saved results. `provenance/execution_manifest.json` seals the exact physical execution sources and binary; `provenance/portability_equivalence.json` documents a metadata-only portability cleanup with identical selection cost coefficients. Native source comments and metric descriptions were corrected without changing the executed binary or gate thresholds.
