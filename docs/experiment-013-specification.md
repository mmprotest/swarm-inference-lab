# Experiment 013: Persistent Event-Driven Subtree Collectives

## Status

This is the preregistered experimental specification for Experiment 013. It is
written before the Experiment 013 execution path is implemented. Raw artifacts,
including unsuccessful trials and rejected redesigns, are retained under the
single immutable run root recorded in the final report.

## Research question

Can a preinstalled, persistent, event-driven, worker-owned subtree collective
reuse worker execution machinery across repeated inference operations while
preserving bounded root coordination, exact deterministic reduction, fault
recovery, recursive cancellation, and the existing model-family boundary?

The experiment is a same-machine control-plane study. It does not measure a
physical LAN or WAN and it does not convert synthetic operations per second into
model tokens per second.

## Prior evidence and limiting mechanism

Experiment 012 established genuine worker-owned delegation through 1,000
independent worker processes. At branch-factor saturation the root performed 16
messages and eight waits, root-to-leaf RPCs remained zero, total messages were
exactly `2N`, and deterministic reduction, retry, cancellation, and the supported
real-model path passed. Its N=1000 p50 was 610.8157 ms and throughput was 1.6372
operations/s.

The Experiment 012 traces and implementation show that persistent TCP sessions
removed connection establishment but did not remove per-operation activation:
each intermediate node created and joined fresh child-dispatch threads and the
root created a fresh executor for every operation. The H012-004 connection-reuse
cycle improved N=1000 p50 by only 0.4%. Experiment 013 therefore changes worker
execution lifetime, not merely connection lifetime.

## Primary hypothesis

**H013-PRIMARY.** Once a collective is READY, persistent worker receive,
dispatch, execute, and reduce loops can remove topology reconstruction and task
creation from the warm critical path. Warm latency scaling will materially
improve from Experiment 012's preferred `N log2 N` model while bounded root work
and exact behavior remain intact.

## Secondary hypotheses

| ID | Falsifiable prediction |
| --- | --- |
| H013-001 | Persistent execution loops reduce worker/task creation per warm operation to zero. |
| H013-002 | Installed routing reduces topology rebuilds per warm operation to zero. |
| H013-003 | Event-driven mailboxes materially reduce scheduler activation relative to Experiment 012. |
| H013-004 | Reusable envelopes or operation slots reduce measured allocation/serialization cost without changing outputs. |
| H013-005 | A model simpler than `N log2 N` best describes warm latency, with materially lower absolute latency. |
| H013-006 | Root degree, waits, messages, and bytes remain bounded through N=1000; root leaf RPCs remain zero. |
| H013-007 | Reordering, duplicates, stale generations, delayed responses, tree-shape changes, and long sequences remain deterministic. |
| H013-008 | Bounded transient retry recovers without contaminating the next generation; permanent loss fails closed or triggers an explicit rebuild. |
| H013-009 | Cancellation reaches the live subtree recursively and cannot contaminate the next operation. |
| H013-010 | The supported real-model validation path retains token equality and tensor-level agreement. |
| H013-011 | A measured sequence length amortises collective setup relative to repeated Experiment 012 delegated operations. |
| H013-012 | N=73 warm control-plane overhead is small enough to justify the next, real Kimi K3 benchmark stage. |

## Architecture under test

Each worker remains an independent process and communicates over framed TCP,
including on the single development machine. The root installs a signed route and
parent/child ownership once. Each worker then starts a bounded mailbox-driven
execution loop and persistent child-dispatch machinery. A warm operation carries
an operation ID, monotonically validated generation, deadline, and the minimum
operation payload. Static topology is read from installed worker-local state.

Intermediate workers, never the root, own child dispatch, response collection,
deterministic child-order reduction, bounded retry, and recursive cancellation.
No shared-memory cross-process shortcut is permitted. Root-to-leaf RPCs must stay
at zero on delegated topologies.

## Experimental loop

Every retained or rejected design follows:

`Hypothesis -> implementation -> benchmark -> result inspection -> bottleneck -> redesign`

Cycle 1 will use the simplest genuine persistent mailbox architecture. Later
cycles are selected only from measured bottlenecks. At least one post-cycle-1
redesign is required even if the initial gates pass. Failed approaches remain in
the cycle ledger and raw evidence.

## Immutable Experiment 012-style baseline

Before modifying the execution path, reproduce the existing delegated-parallel
path with this preregistered matrix:

| Parameter | Value |
| --- | --- |
| Worker counts | 2, 8, 32, 73, 128, 512, 1000 |
| Network profile | `same_host_shaped` |
| Branch factor | 8 |
| Maximum root concurrency | 8 |
| Payload | 256 bytes |
| Warmups | 1 per cell |
| Measured trials | 5 per cell, reduced only for evidenced Windows resource failure |
| Operation deadline | 30 seconds |
| Startup deadline | 180 seconds |

The raw baseline directory is immutable after the run. Median, p50, p95, p99,
dispersion, successful/failed count, and trial count are derived without deleting
failed attempts. A five-trial p99 is reported as indicative only.

## Persistent benchmark matrix

Serious candidates are measured at N=2, 8, 32, 73, 128, 512, and 1000. Cold
results include worker startup, topology construction, route installation,
persistent loop creation, endpoint establishment, and readiness. Warm results
exclude those setup costs and use the already-live collective.

Sequence lengths are 1, 2, 8, 32, 128, and 512 operations through one collective.
N=73 is a first-class cell. Long sequences provide the statistically useful tail
sample; small baseline samples are not used for strong p99 claims.

Branch factors 4, 8, 16, and 32 are compared at discriminating scales subject to
measured host resource limits. The production planner remains conservative; no
universal default is inferred from same-host evidence.

## Instrumentation contract

Cold lifecycle telemetry records setup and teardown time, process and persistent
task starts/stops, topology construction, route installation, endpoint/session
establishment, reusable allocation, reset count, and readiness.

Every operation records submission, root dispatch, intermediate dispatch, leaf
execution, reduction, response propagation, completion, queue delay, scheduler
delay, serialization/deserialization time where measurable, application-level
task/process wakeups, activations, newly created tasks/connections, and practical
allocation counters. Root CPU and worker CPU are recorded separately.

Root metrics are RPCs, messages, bytes, waits, degree, leaf RPCs, and CPU. System
metrics are messages, bytes, RPCs, connection activity, worker CPU, scheduler
events, depth, latency, and throughput. App-level wakeup counters are not claimed
as operating-system scheduler context-switch counters.

Memory is sampled before and after sequences. Resource leakage is assessed from
RSS growth, live task/thread/process counts, connection counts, and successful
teardown.

## Correctness and fault protocol

The test matrix includes reordered arrivals, duplicate requests and responses,
stale and delayed generations, bounded retry, multiple tree shapes, long
sequences, timeout, transient child failure, supported parent/intermediate
failure, cancellation, and permanent loss. A failed generation must not alter the
next generation. Irrecoverable loss must fail closed; rebuilding must be explicit
and its cost reported.

The supported real-model path compares exact selected tokens, tensor tolerances,
cosine similarity where applicable, root work, latency, and traffic. It is not a
Kimi K3 execution claim.

## Scaling analysis

Warm p50 latency candidates are fit to constant, `log2 N`, `N`, and `N log2 N`
models. Fits report coefficients, residual error, R-squared, AIC/AICc where
defined, sample count, and uncertainty/limitations. The preferred model is chosen
by information criterion, not visual preference. Absolute and relative changes
against the immutable baseline are both reported.

## Critical acceptance gates

The thesis passes only if all gates pass:

1. No warm topology rebuild, process creation, persistent-loop creation, or new
   reusable-session establishment.
2. Bounded root metrics after branch-factor saturation and zero root leaf RPCs.
3. At N=1000, at least 50% lower warm p50 than 610.8157 ms and at least 2x the
   1.6372 operations/s reference throughput.
4. Warm scaling materially improves from the Experiment 012 `N log2 N` result.
5. The full required N=73 cold, warm, traffic, activation, depth, and break-even
   record is present.
6. Deterministic state-isolation tests pass.
7. Fault recovery and recursive cancellation pass with measured costs.
8. The existing supported real-model path retains exact behavior.
9. Unit/integration, Experiment 012/013 regressions, Ruff, Mypy, packaging, and
   artifact validation pass before canonical promotion.

Canonical promotion is conditional on these gates. Flat fallback, the Experiment
012 delegated fallback, explicit planner selection, signed route validation,
deterministic reduction, parent retry, recursive cancellation, separate root and
system telemetry, and the WAN boundary must remain available.

## Reporting discipline

The final evidence bundle contains the cycle ledger, raw records, traces,
machine-readable summary, acceptance-gate decisions, Markdown reports, standalone
HTML, validation receipt, and publication-quality figures. Setup cost, tail
latency, failures, resource limits, and rejected designs are retained. Measured
facts are separated from Kimi K3 projections.

