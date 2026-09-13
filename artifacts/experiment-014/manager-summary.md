## Verdict

* Experiment 014 pre-cluster certification: **PASS**
* Complete Kimi CUDA graph: **PASS**
* RTX 3090 sm_86 package: **READY** (physical execution remains NOT RUN)
* Canonical persistent Kimi runtime: **PASS**
* Safe certified batch: **8** (first rejected batch: 9, before CUDA)
* Sub-layer microwork: **FUNCTIONAL BUT NOT CURRENTLY ECONOMIC**
* Smallest useful worker VRAM: **8 GiB** for an optional four-way expert partition inside a fast domain
* Recommended fleet size: **93**
* Recommended topology: **WHOLE-LAYER**
* Maximum worker VRAM: **22.609 GiB planned on 24 GiB**
* Minimum worker headroom: **3.791 GiB total; 1.391 GiB remains beyond the 10% safety reserve**
* KDA p50: **2.975 ms** (production batch 1)
* MLA p50: **2.704 ms** (production batch 1)
* Sub-layer distributed layer p50: **4.189 ms**
* Real coarse boundary bytes: **258,048 payload / 258,283 mean wire**
* Real microwork bytes: **289,550.6 mean total / 100,352 critical path per token**
* Required coarse network: **<= 5.0 ms RTT and >= 10.0 Gbps**
* Required microwork network: **<= 0.5 ms RTT and >= 2.5 Gbps**
* Capacity model held-out error: **0.553% local median APE**
* Projected RTX 3090 capacity retention: **39.434% wall**
* Projected aggregate output throughput: **97.150 tok/s**
* Projected per-user decode: **0.950 tok/s** at the admitted coarse edge
* Fleet cost/hour: **$4.65** at $0.05/GPU-hour
* Cost/M output: **$13.296**
* Margin at $15/M: **11.363%**
* Remote distribution: **PASS**
* Bootstrap: **PASS** locally / physical clean node NOT RUN
* Recovery: **PASS**
* Final rehearsal: **PASS**
* Experiment 015 package: **READY**
* Ready to rent GPUs: **YES — rent the single RTX 3090 canary first; full-fleet activation remains locked**

## Required sub-layer summary

* Sub-layer microwork execution: **FUNCTIONAL BUT NOT CURRENTLY ECONOMIC**
* Smallest tested worker footprint: **0.983 GB (0.915 GiB)**
* Fraction of complete layer per smallest microworker: **5.377%**
* Best sub-layer worker count per layer: **4**
* Sub-layer layer-throughput relative to one-GPU baseline: **78.809%**
* Sub-layer capacity retention: **84.153%** at logical batch 8
* Maximum viable microwork RTT: **0.5 ms tested** (0.646 ms exact at 100 Gbps)
* Minimum viable microwork bandwidth: **2.5 Gbps tested** (1.986 Gbps exact at 0.25 ms)
* Expert-routing imbalance: **6:2 hottest:coldest selections (3.0x)** in the retained three-position trace
* Recommended use of microworkers: **ONLY INSIDE FAST DOMAINS; not in the initial fleet**
* Experiment 015 topology: **WHOLE-LAYER**

The initial fleet is economically viable only near the measured low-price case. The modeled break-even GPU price is approximately **$0.056/hour** at full modeled utilization and a $15/M selling price; the $0.08/hour and higher cases lose money. Sequential dependency depth also limits one stream to 0.950 tok/s even though aggregate batch-8 capacity is 97.150 tok/s.
