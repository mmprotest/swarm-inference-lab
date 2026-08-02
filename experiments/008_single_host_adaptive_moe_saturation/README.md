# Experiment 008: Single-Host Adaptive MoE Saturation

Experiment 008 measures whether one Windows CUDA workstation can execute a sparse MoE whose
weights exceed physical VRAM, and whether capability-aware tensor placement improves on a fairly
tuned llama.cpp baseline. Missing backend hooks are reported as unsupported; fixture evidence is
never promoted to the official verdict.

Quick software and hardware validation:

```powershell
.\experiments\008_single_host_adaptive_moe_saturation\reproduce.ps1 -Quick
```

Official real-model run (downloads the pinned llama.cpp release and preferred GGUF when absent):

```powershell
.\experiments\008_single_host_adaptive_moe_saturation\reproduce.ps1 -Full
```

Use `-ModelPath` and `-ServerPath` for pre-provisioned artifacts, `-SkipDownload` for an offline
run, `-OutputDirectory` for an explicit resumable run parent, and `-Resume` to continue it. The
bundle is written below an `experiment_008` directory and contains raw JSON/CSV evidence, logs,
profiler traces, plots, `report.md`, and `verdict.json`.

Only `-Full` can satisfy official gates. `-Quick` runs real PyTorch tensor kernels and a deterministic
tiny MoE fixture, but all model-behaviour evidence is tagged `EMULATED`.
