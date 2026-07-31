# Experiment 004: Production-Speed Qwen3 Stage Engine

## Scope and immutable baseline

Experiment 004 asks how close the custom Qwen3 stage engine can get to a current
production inference engine on the same RTX 5090 while preserving exact greedy
tokens and a stage boundary that remains usable by future remote workers.

The pre-change source baseline is commit
`49b736acc89f32eb7f647144d4440cbc20eceee3` (branch `main`, commit timestamp
2026-07-31T06:47:05+10:00). The worktree was already dirty before Experiment
004 began: 2,361 tracked entries under `artifacts/` were deleted. Those
pre-existing deletions are not restored or treated as Experiment 004 changes.

The pre-change environment is Windows, Python 3.11.9, PyTorch 2.13.0+cu130,
Transformers 4.57.6, CUDA runtime 13.0, NVIDIA driver 591.86, and an RTX 5090
(compute capability 12.0, 32,607 MiB reported VRAM). Nsight Systems is not
installed. Nsight Compute 2025.3.1 is installed. WSL2 is installed, with
`docker-desktop` as the default distribution.

The untouched test suite completed with 119 passed and 2 explicitly skipped in
35.56 seconds. The first invocation ran every test successfully but pytest
failed while cleaning an ACL-inaccessible `%TEMP%\pytest-current` symlink. A
second invocation with an experiment-local `--basetemp` completed normally.
Both logs are retained under
`artifacts/benchmarks/experiment-004/prechange/`.

## Execution profiles

Every engine record, log line, result row, and report section identifies one of
these profiles:

* `qwen3_correctness` is the retained deterministic oracle. It permits eager
  attention, `DynamicCache`, intermediate boundary checks, NumPy transport, and
  diagnostic full logits.
* `qwen3_fast` is the serving path. It uses GPU-resident tensors at local
  boundaries, a preallocated cache, measured attention-backend selection,
  final-stage sampling, true batched forwards, and an iteration scheduler.
  Boundary diagnostics and full-logit return are opt-in.

Performance evidence is invalid if a row labelled `qwen3_fast` silently falls
back to `qwen3_correctness`. A fallback is instead emitted as a failed
optimisation-ladder row with its diagnostic.

## Existing hot path

### Coordinator and worker path

For every generated token, `CoordinatorService`:

1. Builds a new Python `token_ids` list and a new NumPy `int64` activation.
2. Constructs an `ActivationTensor`, canonical JSON header, raw byte payload,
   and SHA-256 checksum.
3. Appends a replay-log entry.
4. Performs a coordinator-to-worker RPC or persistent-stream send.
5. The worker validates metadata and checksums, parses JSON, creates a NumPy
   view, then copies it.
6. `ExecutionEngine` calls `Qwen3StageModule.execute()` once per queue member.
7. The worker converts output to bytes, hashes it, signs metadata, and returns
   it. Direct routes additionally decode and re-encode the tensor between
   workers.
8. The coordinator decodes the final full-vocabulary vector, casts/views it as
   FP32, calls NumPy `argmax`, computes a diagnostic top ten with
   `argpartition`/`argsort`, tokenises each diagnostic token for display, and
   appends Python metric dictionaries.

The final stage returns `[batch, 1, vocabulary]` logits. For Qwen3-0.6B that is
151,936 BF16 values before the current final-stage FP32 conversion: 303,872
bytes in BF16 or 607,744 bytes in FP32 per generated token, before framing,
checksums, signatures, and protocol metadata.

### Stage module

`Qwen3StageModule.forward()` performs these operations on every invocation:

* embedding stages call `inputs.to(device)`, even though `execute()` has just
  copied the same input to that device;
* a five-element Python tuple is built for the cache key and a dictionary
  lookup is performed;
* a `StageLocalKVCache` and Transformers `DynamicCache` are constructed on the
  first token;
* `torch.arange()` constructs position IDs and another view is made for
  `cache_position`;
* RoPE position embeddings are recomputed;
* `_causal_mask()` allocates two `arange` tensors, a comparison tensor, scalar
  zero and minimum tensors, a `where` result, two added dimensions, and an
  expanded four-dimensional mask;
* every layer iteration constructs a global integer, decimal string, and
  `ModuleDict` lookup;
* `inspect.signature(layer.forward)` reflects on the bound method;
* a new keyword dictionary is built and populated;
* cache-argument spelling is checked by string membership;
* tuple-versus-tensor output is tested;
* the last-layer activation is detached and retained;
* Python cache position validation and mutation runs;
* final norm and `lm_head` run; during prefill the output projection is applied
  to every prompt position even though `execute()` later retains only the last
  logit row.

### Existing cache

The cache is a Python dictionary keyed by request ID, model revision, stage ID,
route generation, and cache generation. Each value wraps Transformers
`DynamicCache`. Dynamic cache storage grows through Transformers cache-layer
updates. Cache accounting walks Python layer objects and tensors. Cancellation
and replay scan dictionary keys, create summaries, walk cache tensors, and
append history dictionaries. This design is valuable evidence for correctness
and replay, but it is not a fixed-address decode cache.

### Existing batching

`ExecutionEngine._run()` waits for nominally compatible queue entries and puts
them in a Python `batch` list. It then loops over that list and calls
`_execute_one()` separately for every request. Each member therefore gets a
separate tensor decode, host-to-device copy, model forward, device-to-host
copy, serialisation, signature, and event-loop yield. There is no batched model
tensor or shared forward.

## Synchronisation inventory

The normal current stage path has these explicit CUDA synchronisations:

1. after host-to-device activation copy in `execute()`;
2. after the model forward in `execute()`;
3. after the prefill boundary diagnostic call, even when no boundary reference
   is configured;
4. after device-to-host output copy in `execute()`.

There is also a synchronisation after stage weight loading and one after
stage-local warm-up. Reference generation contains separate correctness-only
synchronisations. Implicit synchronisation occurs when a CUDA tensor is copied
to CPU and when scalar results are materialised with `.item()`. The fast path
must contain no `torch.cuda.synchronize()` outside an explicit measurement,
graph-capture, transport-boundary, or correctness boundary. GPU timing uses
CUDA events.

## Host/device copy inventory

For every current stage invocation:

* NumPy input is made contiguous and, for token IDs, viewed/cast as `int64`;
* `torch.from_numpy()` creates a CPU tensor view;
* `.to(device)` copies token IDs or BF16 hidden-state bits to CUDA;
* every stage output is detached and copied to CPU;
* intermediate BF16 is viewed as `uint16` and exposed as NumPy;
* final logits are converted to FP32 and exposed as NumPy;
* tensor framing makes contiguous NumPy views and copies raw bytes;
* decoding normally copies the NumPy buffer again.

Boundary validation adds a BF16/FP32-to-FP32 CPU copy and NumPy conversion at
prefill boundaries when enabled. Direct worker forwarding decodes and re-encodes
the same bytes. In a one-worker direct route the coordinator still serialises
the input and receives/decodes the output.

## Dynamic-allocation and Python-operation inventory

The measured decode loop currently allocates or constructs:

* coordinator lists, NumPy arrays, tensor envelopes, canonical JSON, byte
  payloads, hashes, replay entries, metadata copies, timing dictionaries, and
  top-logit diagnostics;
* worker decoded arrays, queue items/futures, operation/detail dictionaries,
  output envelopes, hashes, signatures, and metric records;
* stage contiguous arrays/tensors, position tensors, a causal mask and its
  temporaries, per-layer signature objects and keyword dictionaries,
  output NumPy arrays, and transfer/history dictionaries;
* multiple `time.perf_counter()`/`monotonic_ns()` calls, enum-to-string
  conversions, dictionary/list mutations, type/shape checks, and event-loop
  transitions.

The exact cache allocation happens only at request start, but `DynamicCache`
updates and Python cache dispatch remain inside every layer call.

## Profiling protocol

The unmodified one-worker path is measured with three warm-up requests, five
complete 512-token repeats, CUDA synchronisation counting, process CPU time,
transfer counters, PyTorch CPU/CUDA profiling, and a Chrome trace. The top ten
operators and the measured throughput are written to
`artifacts/benchmarks/experiment-004/prechange/legacy-baseline.json`; the trace
is `legacy-baseline-trace.json`. Experiment 004 adds conditional NVTX ranges
for:

`tokenisation`, `request_admission`, `embedding`, `position_setup`,
`attention_mask_setup`, `decoder_layer`, `final_norm`, `lm_head`, `sampling`,
`cache_update`, `host_to_device`, `device_to_host`, `tensor_encoding`,
`tensor_decoding`, `grpc_send`, and `grpc_receive`.

NVTX instrumentation is disabled in unprofiled fast runs so it is not itself a
headline-path cost.

### Measured pre-change result

The retained in-process `Qwen3StageModule.execute()` path measured a median
38.2746 decode tokens/s (minimum 38.2418, maximum 38.4571, CV 0.201%) across
five complete 512-token repeats. Each repeat made 1,537 explicit
`torch.cuda.synchronize()` calls. The profile request issued 36,924
`cudaLaunchKernel` calls for 16 decode tokens, or approximately 2,308 launches
per token including profiler-visible runtime calls. The largest profile
contributors were the enclosing legacy decode step, CUDA kernel launches,
matrix multiplies, elementwise multiply/copy/add, matmul dispatch, fills, and
empty-strided allocation.

The actual one-worker process/RPC serving path measured a median 28.4953 output
tokens/s (minimum 28.4533, maximum 28.6437, population standard deviation
0.0685, CV 0.240%) across five complete 512-token requests. Median per-token
decode latency was approximately 34.0 ms. All five requests matched the
retained reference, the direct data plane was used, and the worker shut down
cleanly. This process-path value is the Experiment 004 acceptance denominator:
the fixed 4x gate is therefore 113.9810 verified output tokens/s. The direct
module number is retained to distinguish model/stage costs from process and
transport costs.

## Optimisation ladder

The ladder is applied and measured in this order:

1. Preserve the legacy NumPy/eager/`DynamicCache` path as
   `qwen3_correctness`.
2. Bind layer modules, cache argument style, and forward closures once at load;
   reuse position and causal-mask storage.
3. Add GPU-native single and batched stage calls; retain activations on CUDA
   for in-process stages.
4. Select `eager`, `sdpa`, `flash_attention_2`, or `flashinfer` only after
   availability, startup correctness, and timing checks. Unsupported explicit
   selections fail clearly.
5. Add a stage-local contiguous static cache with fixed capacity, in-place
   append, request-slot ownership, exact cleanup, snapshots/forks, rollback,
   byte accounting, and fragmentation reporting.
6. Sample on the final worker and return token ID plus selected logit; retain
   optional full logits only in diagnostic mode.
7. Compile prefill and decode separately, recording cold compile cost, graph
   breaks, fallbacks, and steady-state timing for default, reduce-overhead, and
   max-autotune modes.
8. Attempt verified manual CUDA graph capture/replay for fixed batch buckets
   1, 2, 4, 8, 16, 32, and 64. A failed or slower capture remains an explicit
   failed ladder row. Before capture, conservatively project retained
   position-specialised graph tensors, KV capacity, and model storage against
   an 88% device-memory budget. Reject the bucket with a recorded fallback
   reason when capture would force CUDA/WDDM paging; continue with the exact
   eager/dynamic current-length cache path for that shape when the static-view
   allocator footprint would also exhaust the device.
9. Replace nominal queue batches with one tensor forward and per-request cache
   slots; report useful tokens, padding, formation wait, and forward duration.
10. Drive batches with an iteration scheduler using `latency`, `balanced`, and
    `throughput` policies.
11. Compare in-process GPU references, same-host pinned BF16 transfer, and
    remote-compatible pinned BF16 binary framing separately.

## Correctness risks

* SDPA and FlashAttention change reduction order. Logits need not be
  bit-identical, but every greedy token must match the deterministic oracle.
* Static cache positions can be off by one at the prefill/decode transition,
  can expose uninitialised capacity to attention, or can cross-contaminate
  request slots.
* Batched position IDs and masks can broadcast incorrectly, especially when
  requests have different sequence lengths.
* Padding can alter attention unless padded KV slots are fully masked.
* Graph replay can capture stale tensor addresses, cache positions, or batch
  membership.
* A graph bundle can technically fit while exhausting dedicated VRAM and
  paging into host memory. Admission therefore needs a projected-memory guard,
  not only an out-of-memory exception.
* Compilation can silently graph-break and appear enabled without executing a
  compiled region.
* Global deterministic/CUBLAS flags set by the correctness module can leak into
  the performance profile.
* Tied input/output embeddings must preserve parameter identity in a
  single-stage layout.
* Sampling on the worker can change tie handling, repetition penalties, RNG
  streams, or seed semantics.
* BF16 transport must preserve raw bits and byte order; conversion through FP32
  is not equivalent evidence.
* Asynchronous pinned-memory copies need an explicit event/stream ownership
  hand-off at a real process or network boundary.
* CUDA IPC is not assumed safe on Windows/WDDM. If unavailable, same-host
  process evidence must identify pinned-host staging instead.
* A fast monolithic one-process result does not establish remote multi-stage
  production speed. Remote transport remains a separately measured path.

## Acceptance interpretation

The minimum performance gate is fixed at 4.0 times the remeasured legacy
custom baseline and at least 50% of the fastest successful production engine,
with coefficient of variation at most 10%. The 80% production-parity target is
reported separately and does not change the minimum gate. Missing required
SGLang or Qwen3-4B coverage is an explicit failed status, never a substituted
model or silently omitted row.
