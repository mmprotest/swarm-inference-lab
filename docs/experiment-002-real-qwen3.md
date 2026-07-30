# Experiment 002: real process-isolated Qwen3 inference

## Classification and claim

Experiment 002 runs in `single-host-loopback-real-model` mode. It proves that one real
Qwen3 checkpoint can be partitioned into independently loaded, process-isolated stages;
that hidden-state tensors move directly from worker to worker; that each worker owns the
KV cache for its decoder layers; and that greedy output is token-identical to an
independent full-model process.

It does not prove multi-machine, LAN, WAN, Raspberry Pi, or Kimi K3 execution. Four
workers on one RTX 5090 do not add physical compute, and this experiment makes no
single-request speedup claim.

## Baseline inspection

Before Experiment 002, the repository already provided:

- a fail-closed Qwen3 tensor-name adapter;
- safetensors header inspection without GPU model construction;
- target-byte-based contiguous shard construction;
- partial Qwen3 stage construction without
  `AutoModelForCausalLM.from_pretrained()` in a worker;
- a direct worker-to-worker gRPC data path with signed route generations,
  checksummed binary tensors, and persistent peer streams;
- coordinator replay input logging and route recovery;
- a separate-process full-model correctness helper; and
- synthetic experiment status and reporting.

The baseline did not satisfy this experiment because:

- `shard-model` did not require exactly four stages;
- the shard artifact did not contain the complete Experiment 002 manifest, hashes,
  config, tokenizer, stage metadata, and validation layout;
- validation could co-load every split module in one process;
- BF16 activations were converted to FP32 for transport;
- worker load proofs omitted actual PID, RSS, CUDA, shard, and tensor evidence;
- stage-local cache evidence did not expose global/local layer mapping and shapes;
- replay replaced or rebuilt more state than one selected live stage;
- prompt-suite, concurrency, exact mismatch diagnostics, boundary checks, and
  fail-closed real-model statuses were incomplete; and
- the launcher and required evidence bundle were absent.

The untouched relevant baseline executed 12 tests successfully. Pytest then encountered
a host-only permission error while cleaning its default Windows temporary symlink. All
subsequent test runs use repository-local temporary directories. The baseline also
confirmed the earlier dependency regression: the project venv lacked PyTorch. CUDA
PyTorch was restored additively with `uv pip install`; no sync or pruning command was
used.

## Immutable model inspection

The default and only implicit model is `Qwen/Qwen3-0.6B`. Hugging Face resolution must
produce an immutable commit. A different model is accepted only when explicitly
selected, has the supported dense `Qwen3ForCausalLM` architecture, and contains fewer
than one billion parameters.

Inspection reads `config.json`, safetensors indexes, and safetensors headers. It does not
instantiate the complete model or place weights on a GPU. Evidence includes:

- requested ID, immutable revision, and local snapshot;
- config, tokenizer, and safetensors file hashes;
- architecture dimensions and RoPE settings;
- tensor count, parameter count, dtype, and tied-weight state;
- source, embedding, output-head, final-normalisation, and per-layer bytes;
- largest layer;
- boundary activation estimates; and
- per-token, per-layer KV-cache estimates.

Unsupported and unmapped tensors are fatal.

## Four-stage partition

The exact-stage partitioner first finds the minimum possible maximum stage weight, then
minimises squared distance from the ideal stage byte count under that cap. Every
candidate cost uses actual source tensor sizes. Stage 0 includes embedding ownership;
the final stage includes final normalisation and output projection ownership.

Qwen3-0.6B declares tied input and output embeddings, while this immutable source
checkpoint contains both endpoint tensor names and counts both byte ranges in the source
safetensors. Experiment 002 assigns the embedding tensor to stage 0 and the explicit
LM-head tensor to stage 3, without creating another shard-time copy. The manifest records
this source treatment, both owners, zero additional duplicated bytes, source bytes,
total sharded bytes, and tensor-to-stage mapping. No intermediate worker receives either
endpoint tensor.

Every decoder layer belongs to one contiguous interval and exactly one stage. The
builder verifies the generated safetensors union, declared duplicates, shapes, shard
hashes, and stage count before publishing the manifest.

## Stage-only construction and strict loading

`Qwen3StageModule` constructs only:

- its globally indexed `Qwen3DecoderLayer` interval;
- token embeddings on stage 0;
- final RMS normalisation and vocabulary projection on stage 3; and
- the small rotary-position component required by its local layers.

Workers never call a full-model `from_pretrained()` path. Loading is strict against the
stage manifest: undeclared, missing, shape-mismatched, source-dtype-mismatched, and
incorrectly tied tensors are fatal. Parameters are inference-only and gradients,
optimisers, compilation, CUDA graphs, speculation, quantisation, and sampling are
disabled.

## Attention, positions, and KV cache

Every stage receives the same deterministic token position and sequence metadata. It
uses eager attention, an explicit causal mask, global position IDs, and
`cache_position`. Unpadded batch-size-one prompts avoid cross-stage padding ambiguity.

Qwen3 rotary frequencies are constructed on CPU and then moved to the worker device,
matching the official full-model construction order. Direct CUDA construction changed
two float32 frequency values by one ULP on this platform; the phase error became visible
at 512 tokens even though short prompts and greedy tokens still matched. Deterministic
cuBLAS workspace selection and full-precision BF16 reductions are also applied
identically in the reference process and stage workers.

`StageLocalKVCache` wraps the installed Transformers `DynamicCache` while retaining
global decoder indices. A stage beginning at layer 7 therefore updates cache layer 7,
not cache layer 0. Evidence exposes:

- request, immutable model revision, stage, route, and cache generations;
- global and local layer indices;
- sequence length;
- key/value shapes and dtype;
- initialised and owned layer counts; and
- exact allocated tensor bytes.

Prefill creates cache state and decode appends one position. Completion and cancellation
delete cache state. Completed cache summaries remain as evidence, while live cache count
must return to zero.

For replay, each worker retains its own checksummed stage-input log. The selected
intermediate worker clears only its local KV cache, re-executes its recorded stage
inputs, and resumes the unchanged route. Other stage caches are not reset and the
coordinator never stores a full-model KV cache.

## Direct tensor data plane

The route is:

```text
coordinator -> stage 0 -> stage 1 -> stage 2 -> stage 3 -> coordinator
```

Only prompt or committed token IDs travel from the coordinator to stage 0. Stage
boundaries use the existing non-pickle binary tensor envelope with identifiers, route
generation, operation, position, dtype, shape, strides, byte order, payload length,
sequence number, and SHA-256 checksums.

BF16 hidden states are copied from CUDA to CPU and transported as their raw 16-bit bit
patterns. The receiver reconstructs BF16 before copying to CUDA. This is host-staged
loopback transfer, not GPU peer-to-peer transfer. The final worker sends only the last
position's real vocabulary logits to bound payload size; the coordinator never receives
intermediate hidden states.

PASS requires direct mode, zero coordinator activation bytes, positive worker-to-worker
activation bytes, at least three persistent peer streams, and positive peer send/receive
counts. Relay fallback is disabled and cannot silently satisfy the status.

## Correctness oracle

The reference is a separate process and separate phase. It alone loads the complete
checkpoint with `AutoModelForCausalLM`, eager attention, BF16, the same immutable
snapshot, the same prompt token IDs, raw completion semantics, greedy argmax, EOS
handling, and thinking disabled. The reference process exits before stage workers
start.

The primary criterion is exact equality of the complete generated token-ID list for
every request. A mismatch records its first position, both token IDs and strings,
selected logits, and top logits.

Reference hooks capture the decoder outputs at all four partition boundaries. Workers
compare their own prefill boundary output locally against the reference tensor and
report maximum and mean absolute error, maximum relative error, cosine similarity,
shape and dtype identity, and NaN/Inf counts. Full boundary tensors are temporary and
removed after comparison; the durable bundle stores diagnostics. Tolerances are fixed
by configuration and are never relaxed automatically.

## Prompt suite

The mandatory suite contains:

1. factual completion;
2. arithmetic;
3. code completion;
4. repeated tokens and symbols;
5. punctuation-heavy JSON-like text;
6. exactly 128 input tokens;
7. exactly 512 input tokens; and
8. two concurrent requests.

Each request generates up to 16 tokens by default. After the basic suite passes, a
separate request clears and replays stage 1 after four committed outputs, then continues
generation and requires exact reference identity.

## Memory and isolation accounting

The logical worker weight limit is calculated after partitioning. It must exceed the
largest valid stage plus safety allowance and remain below the full source model bytes.
Workers enforce it before shard construction/loading. A separate logical total-memory
estimate includes module/KV allowance and CUDA-context safety.

Every worker signs a proof containing PID, stage and layer range, shard path/hash,
actual loaded tensor names, loaded weight bytes, endpoint ownership, pre/post RSS,
pre/post/peak CUDA memory, full model bytes, logical limit, and timestamp. The
coordinator proof states zero loaded model-weight bytes, zero intermediate activation
bytes, tokenizer/config/manifest presence, final-logit and control bytes, and host
memory.

PASS requires four verified proofs, one stage per process, exact layer coverage, no
worker with the full tensor set or all decoder layers, a full model larger than every
logical worker weight limit, and zero coordinator model weights.

## Artifacts and fail-closed status

The launcher writes the requested and resolved configurations, environment and git
state, model inspection and manifests, independent reference, distributed results,
load/coordinator/transport/cache proofs, prompt JSONL, replay evidence, events,
boundary diagnostics, seven charts, `summary.json`, non-empty structured worker logs,
quality-gate evidence, log validation, and a self-contained HTML report. Ephemeral
worker signing keys are deleted after proof verification and are never retained in the
evidence bundle.

`overall_status` is PASS only when every mandatory environment, revision, sharding,
isolation, real execution, direct transport, cache, boundary, token, replay, and prompt
status is PASS. Synthetic execution, relay mode, missing proof, token mismatch,
boundary mismatch, missing replay, fatal logs, or stale workers fail closed.
