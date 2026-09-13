# Measurement and replay definitions

- **Physical machine:** one RTX 5090. A global asynchronous GPU lock covers
  every target, draft and state-changing native call. Three target workers and
  the MTP worker retain model state throughout collection. Loading is excluded
  from steady-state decode, and prefill is recorded separately.
- **Virtual inventory:** three independent target GPU service resources and
  one independently charged coordinator/drafter resource. MTP service was
  measured on the same physical 5090; the virtual drafter therefore assumes
  independent GPU-capable compute, not an unmeasured CPU or free drafting.
  This is a resource assumption, not a physical three-GPU measurement.
- **Stage selection:** synchronized callbacks at llama's existing `l_out`
  markers measure 64 layer costs. A contiguous minimax search chooses two
  boundaries and charges the measured residual final-head cost to Stage C.
  Callback measurements select the split only. Simulator services come from
  the actual isolated stage operations under the final split.
- **K and W:** K draft positions follow the leading token, giving K+1 rows per
  full chunk. Later chunks can be wholly provisional. W counts launched chunks
  awaiting coordinator verification/commit. Traces separately record whether a
  new launch preceded older target completion. The scheduler fills its bounded
  window, verifies the oldest chunk, and refills one slot after acceptance.
- **Verification:** target rows execute scalar llama kernels, coalesced into a
  chunk on the wire. This uses the existing reliable scalar stage path and
  preserves the same numerical execution across W. No fused-batch kernel
  speedup is assumed. MTP autoregresses on provisional hidden states; target
  results alone authorize greedy token commitment.
- **Rollback:** workers share KV prefix storage through generic sequence-copy
  metadata and retain recurrent states through llama's copy-on-write machinery.
  Eighteen checkpoint slots bound the maximum W=16 workload. Each checkpoint
  also owns its small activation input. The coordinator sends all three native
  compound rollback requests concurrently. Each worker restores its boundary,
  replays only the accepted prefix from its cached input, and returns position
  and output-checksum metadata. KV or recurrent context is never sent over WAN.
- **Four activation hops:** coordinator→A, A→B, B→C, C→coordinator. The local
  Python relay transports native messages on loopback; the shaped path inserts
  delay on these logical edges. The virtual edges use the actual serialized
  token/FP32 activation payloads and frame headers. No compression is modeled.
- **Network service:** one-way delay is RTT/2 plus seeded uniform jitter in the
  specified range, plus bytes × 8 / bandwidth. Each directed link serializes
  its bytes, while propagation can overlap. Stages explicitly wait for earlier
  sequence positions if jitter reorders arrivals. A rejection reaches each
  worker only after its control transfer; currently running work is charged and
  queued stale work is discarded after cancellation arrives. The local runner
  drains outstanding calls before issuing restore against its shared physical
  device. The virtual remote queues model cancellation-control propagation and
  independently finishing active operations; this async control overlap is
  modeled, not a physical multi-host correctness or performance measurement.
- **Compute service:** exact observed same-chunk stage service where available.
  Virtual cancellation can allow an operation that was canceled locally to
  start; such work uses an empirical median from the same stage and row count.
  A missing measured shape is an error. No duration is obtained by dividing
  whole-model time, using hardware specifications, or supplying synthetic data.
- **Idle-conditioned serial service:** the first W=1 WAN-60 prediction missed
  the fixed validity threshold at 10.57%. Native serial service increased during
  idle gaps. `serial_idle_profile.json` therefore records 32 real operations per
  stage and WAN profile using the second fixed prompt and seed 280915. These
  raw native service medians replace the no-speculation WAN control's warm
  service in both validation and the sweep. Fresh validation uses the first
  prompt. No whole-run latency or correction multiplier is fitted. Speculative
  services retain their original physical workload timings. The initial failed
  validation is preserved under `archives/validation_attempt_1/`.
- **CPU accounting:** physical decode wall time minus all measured serialized
  native service is retained as a host scheduler/adapter gap. It is charged on
  the coordinator as a measured per-action residual, not fitted to WAN results.
  Validation uses independent workload medians, including this gap; native no-speculation service additionally uses the periodic-service calibration.
- **Throughput:** total committed tokens / total decode time. Decode includes
  drafting, fill/drain, transfers, rejection and rollback. Eight prompts are
  separate single-user runs; pooling them does not model eight simultaneous
  users. Fixed-length generation uses 256 token positions without EOS stopping.
- **WAN wait:** the union of intervals with network transfers outstanding but
  no modeled resource doing work, divided by decode time. Work that is later
  discarded still occupies a resource; read this alongside discard fraction.
- **Stage utilization:** modeled resource busy time / decode time, including
  native host overhead in stage service. It is not CUDA SM occupancy. The
  steady-state interval runs from first commit to last launch, without tuning
  the interval against the result.
- **Discard fraction:** discarded target compute / total target compute.
  Entire invalidated chunks are discarded. For a partially accepted scalar
  chunk, compute is apportioned by row count; prefix replay remains charged.
- **Acceptance:** accepted speculative input positions / launched speculative
  input positions. A known leading target token is excluded from both counts.
- **TTFT and inter-token latency:** TTFT includes serial prefill and drafter
  prefill under the implemented task graph, using mean network jitter for that
  prefill estimate. Inter-token latency follows actual commit times; a batch
  release yields zero intervals within the release. This is disclosed rather
  than spreading a release across invented token times.
- **GPU memory:** whole-device memory is sampled once per second. Per-run
  peaks use samples inside that run's recorded wall interval; no claim is made
  that sampling captures a sub-second allocation maximum. Native stage-specific
  allocation peaks are unavailable. The original raw workload files also retain
  the collector's cumulative high-water value, distinguished during replay.
- **Unavailable metrics:** no CUDA-event-only kernel timer or isolated native
  serializer timer is available. Synchronized host compute and composite native
  serialization/state/output costs are preserved under their actual meanings.
- **Validation:** independent W=1 SERIAL runs at 30/60 ms and SPEC_SYNC K=3 at
  60 ms use actual local high-resolution sleeps dispatched off the event loop.
  `timer_probe.json` records the Windows asyncio timer overshoot that motivated
  this injector choice; no simulator correction factor is used. Every condition must predict wall time within
  10%. A further W=8 shaped run checks functional concurrency and exact tokens.
  This cannot validate physical multi-GPU contention or actual inter-host WAN.
- **Evidence exclusions:** `archives/serial_rollback_attempt/` contains an
  incomplete initial E028 implementation with serial rollback/retransmission.
  It motivated the compound state-local control operation and is excluded from
  the final service pool, workload comparison and verdict. Smoke, forced-draft
  and unit-test traces are likewise excluded from performance predictions.

The pass thresholds in `config.json` were fixed before E028 measurement and
remain unchanged through the protocol repair.
