# Experiment 025: Headline Attempt 005

## Technical summary

Headline Attempt 005 is **INCOMPLETE**. The corrected controller launched only after the live full-fleet balance gate returned `GO`, monitored all backbone groups concurrently, and performed 19 attributable instance replacements without destroying healthy siblings. The fleet reached a peak of **46 simultaneously-live READY roles out of 97 required**. All four mandatory Layer 89 fragment workers became READY on distinct physical machines, but the Layer 89 parent and the complete required fleet did not become READY before the 30-minute acquisition cutoff.

No Kimi K3 inference was attempted: Token 1, Token 2, and the exact public sentence were all **not reached**. Cleanup passed, the independent watchdog stopped only after zero-live verification, and a post-cleanup Vast query confirmed **zero live E025 instances**.

## The corrected lifecycle policy worked, but acquisition did not finish

- `READINESS_MONITORING_STARTED` records 41 backbone groups and four Layer 89 fragment groups under `ALL_GROUPS_CONCURRENT_ROLLING_REPLACEMENT`.
- The controller used a 240-second per-instance no-progress threshold and a 30-minute hard acquisition window with a 15-minute inference/cleanup reserve.
- 17 instance-level no-progress timeouts were recorded and 19 replacement instances were created.
- Machine 57056, the progressing host incorrectly torn down in Attempt 004, was retained through genuine progress in Attempt 005 and reached `WORKER_READY` for `e025-layer-089-sub-02`.
- At the hard deadline, 46 roles were still READY. Global teardown occurred only at the allowed terminal acquisition cutoff because the required 97-role fleet was incomplete.

This is evidence that the corrected dud-isolation policy operated as designed. It is not evidence that the full Kimi K3 path executed.

## Layer 89 fragments reached READY but never executed a token

Each mandatory Layer 89 fragment loaded its assigned expert shard, verified the checkpoint fingerprint, and reported a physical consumer GPU identity. Their machine IDs were 143869, 141803, 57056, 31715, all distinct. Readiness establishes physical capacity and worker preparation in this attempt; because Token 1 was never started, it does not establish Layer 89 participation in a full-fleet forward pass.

## Scope and metric definitions

- **Required role:** one of the frozen 93 backbone stages or four Layer 89 sub-layer workers.
- **Simultaneously-live READY:** a worker's latest canonical state is `WORKER_READY`, with no subsequent matching unhealthy, disconnect, or instance-destroy event at that point in event order.
- **Paid instance:** a unique Vast instance with both ledger `CREATE_CONFIRMED` and canonical `INSTANCE_CREATED` evidence.
- **Targeted replacement:** a canonical `INSTANCE_REPLACED` event for one failed instance group.
- **Attempt duration:** the policy's 1,800-second acquisition window; the first ledger-confirmed paid instance occurred 4.103 seconds after the conservative policy origin.
- **Location:** raw Vast-reported host metadata only. No missing country, city, region, or coordinates were inferred.

## Methodology and retained evidence

The post-run finalizer parsed the canonical event stream in event-ID/monotonic order, validated contiguous unique event IDs, computed the stream SHA-256, replayed worker readiness as a state machine, reconciled unique created and destroyed instance IDs, and independently validated every entry in the append-only lifecycle ledger hash chain. It then joined created instances to the frozen offer snapshot for advertised hardware, network, pricing, and raw location fields. Missing values remain explicitly absent.

The canonical trace contains 2521 events. It has no `TOKEN_EXECUTION_STARTED`, `MESSAGE_SEND_STARTED`, `ROUTE_COMPUTED`, `TOKEN_SAMPLE_STARTED`, or `TOKEN_EMITTED` records because inference never began. No token path or output was reconstructed from timestamps.

## Cost and cleanup

The lifecycle ledger estimates **$24.1386** active rental and **$0.6925** storage, or **$24.8311** combined. These use actual ledger lifetimes and selected advertised rates but are **not a provider invoice**. Completed worker-download events support a separate **$5.9666** lower-bound ingress estimate; incomplete downloads and provider billing counters are not fully observed, so this must not be represented as total actual ingress.

All 65 created Attempt 005 instance IDs have destruction evidence. Cleanup and the independent post-cleanup query both report `zero_live_e025_instances = true`; unrelated instances were not destroyed.

## Limitations and PASS-gate status

Attempt 005 cannot support the E025 headline claim. The complete physical fleet identity was never frozen, full-path model execution did not occur, numerical and two-token stateful correctness were not tested, and no public generation exists. The terminal base-summary exception names Layer 89 parent readiness, but the controlling terminal condition was the wrapper's permitted hard acquisition cutoff; the parent exception is the downstream incomplete-fleet manifestation, not a new Kimi K3 execution defect.

The canonical headline summary correctly remains `INCOMPLETE`. Its full-fleet physical worker count is zero because the implementation freezes that field only after complete acquisition; the 46-role figure in this report is an acquisition-state statistic derived from the canonical event stream, not a replacement PASS metric.

## Recommended next step

Do not weaken any E025 PASS gate and do not portray Attempt 005 as model execution. Preserve this attempt as acquisition evidence. Any future paid attempt should be separately authorized, retain the corrected per-instance lifecycle policy, and first use the Attempt 005 churn record to improve live-offer feasibility and alternate depth without redesigning the 97-role scientific architecture.

## Further questions

- Which Vast offer and bootstrap attributes best predict READY completion within the acquisition window?
- How much alternate depth is required to overcome the observed machine churn while retaining a 15-minute correctness and cleanup reserve?
- Can the controller emit a dedicated `ACQUISITION_HARD_DEADLINE_REACHED` event before teardown so the terminal reason is explicit rather than inferred from the recorded policy deadline and subsequent readiness aborts?
