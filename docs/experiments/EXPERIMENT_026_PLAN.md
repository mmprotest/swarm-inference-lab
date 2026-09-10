# E026_Q27_WAN_SWARM_INTEGRATED_PROOF

Status: PREFLIGHT. No inference result or canonical verdict exists yet.

## Question and fixed inputs

Can the exact Qwen3.8-27B Q4_K_M GGUF execute correctly and interactively across at least three physical machines on genuine WAN links, with useful shard caching and recovery that does not replay the original prompt?

The user-specified gates are retained: median committed decode >=8 token/s, >=1.5x ordinary WAN throughput, synchronization target >=2.5 committed tokens/target traversal, disk-warm startup >=3x faster than uncached, interruption <20 s with no lost/duplicate committed tokens, and Vast spend <=38 USD. One sealed generation must contain >=512 generated tokens. At least one application RTT >=80 ms is preferred and must be measured, never manufactured.

Target source candidate: `ggml-org/Qwen3.8-27B-GGUF`, revision `0669b98607d47046c7c2b3f801011d54a08cfccf`, file `Qwen3.8-27B-Q4_K_M.gguf`. Acquisition must verify upstream SHA256 and inspect GGUF metadata before accepting it. No target or quantization substitution is permitted. Native MTP is a separate experimental arm, with its weights and build independently recorded.

## Reconnaissance and reuse

- Repository baseline: `d4b4cab`; the existing tree is dirty. Preserve all existing edits and all historical artifacts. E026 source, receipts, downloaded runtime, and results have separate paths.
- E018/E019/E021 provide task traces, exactness checks, state ownership examples, and evidence-class discipline. Their Kimi task graphs, giant resource inventories, and modeled service times are not E026 performance evidence.
- E025 provides append-only lifecycle logging, provider balance queries, per-instance readiness monitoring, cleanup receipts, and strict deadlines. Its final attempt never reached inference; acquisition readiness is not a throughput result.
- Canonical runtime has persistent stage transport, TLS, request ordering, GGUF/RPC placement, and metered connections. Existing native stages do not cover this hybrid model. Existing recovery replays the entire prompt and cannot pass E026.
- Upstream llama.cpp hybrid sequence state serializes both attention and recurrent memory when PARTIAL_ONLY is not set. Validate actual export/restore and resumed tokens locally before relying on it.
- Local hardware preflight reports RTX 5090, 32607 MiB, driver 591.86, CUDA capability 12.0. Vast read-only query reports 45.80512448978996 USD credit and no instances. These observations will be captured in dated receipts, not treated as enduring capabilities.

## Sequential experiment

1. Pin and verify source, compiler/build, exact GGUF, local hardware, and budget. Freeze development/held-out corpus with short, approximately 2K, and approximately 8K contexts and six content categories. Keep final prompts unused until configuration freeze.
2. Establish greedy local target reference, 256-token measurements where possible, complete token IDs, timings, numerical validation data, state sizes, and deterministic repeat. Evaluate native MTP at a small depth sequence; reject unstable/divergent configurations. Separately validate complete state restore plus bounded tail replay.
3. Locally validate the distributed path and telemetry on the same target. This is development evidence only. Use coarse contiguous ownership as this experiment's initial choice; it is not a permanent product topology. Boundaries must be derived from real work and memory.
4. Only when local correctness works, rent two heterogeneous remote nodes. Record machine IDs, provider geography and application-level RTT/throughput. Obtain the ordinary uncompressed, nonspeculative WAN baseline before optimization. Keep RPC endpoints behind authenticated encrypted transport.
5. Use measured critical-path decomposition to select one mechanism per causal comparison. Record hypothesis, change, expected outcome, frozen control, quantitative result, and keep/modify/revert/defer decision. Count target traversals, logical WAN traversals, physical commands, and bytes separately. Never infer network wait by summing overlapping stage clocks.
6. Measure uncached, disk-warm, GPU-warm startup using identical residency definitions and per-node model transfer accounting. Add the minimum standby needed for recovery. Kill a production process without warning, restore complete hybrid state, replay only a bounded tail, and validate the entire committed stream against control.
7. Freeze surviving configuration, thresholds, topology/placement, speculation, and recovery policy. Run unused prompts, including >=512 tokens, replicate important positive results, run no-failure/failure controls, then destroy every E026 rental and reconcile cost.
8. Generate auditable run JSONL, token/event artifacts, final receipt, and concise Markdown report with exactly one supported canonical verdict. Missing measurements remain null with reasons. Failed gates are not infrastructure invalidity.

## Initial falsifiable hypotheses

- H1: correct ordinary WAN execution is dominated by serial network waits rather than decode activation bandwidth. Measure actual command/byte traces; defer activation codecs if boundary transfer is immaterial.
- H2: native MTP can commit >=2.5 exact greedy tokens per expensive traversal and improve end-to-end WAN throughput >=1.5x. Target-only execution controls token identity; local MTP speed alone does not establish WAN value.
- H3: cached, stage-owned weights make disk-warm readiness >=3x faster than uncached under measured acquisition conditions.
- H4: complete hybrid state checkpoints or verified state replication allow recovery within 20 s without full prompt replay and without losing/duplicating committed tokens.

## Budget and stopping rules

The absolute autonomous Vast ceiling is 38 USD, including rental, storage, ingress and egress. Available credit must also retain >=7 USD. Provisioning must reserve a conservative bounded lease, model transfers, measured traffic allowance, and teardown margin before creation. Record creation intent before calling the provider and reconcile uncertain creates by E026 label. Use an independent deadline/budget watchdog and verify destruction with provider inventory. Stopped disks are not free; destroy unused instances.

Do not rent during local implementation. Stop unproductive paid arms promptly. Freeze and preserve evidence if the remaining budget cannot fund the next valid trial. Required credentials, actual security blockers, the budget ceiling, or a decisive supported result are the authorized early stops. Unfinished implementation is not a scientific negative result.
