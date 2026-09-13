# E027 native stage service

`e027_stage_server.cpp` is the deliberately small data plane for E027. Each
process loads the fixed GGUF once, owns one contiguous layer interval and one
llama context, and accepts coarse batches over a persistent TCP connection.
Only token IDs (stage 0), FP32 stage-boundary hidden states, and compact final
top-k/hidden results cross process boundaries. No ggml backend operation is
exposed on the wire.

The companion `qwen35-stage-range.patch` is generated from the pinned E027
llama.cpp worktree by `scripts/experiment_027_prepare_llama.py`. It makes the
Qwen35 graph honor `E027_STAGE_START`/`E027_STAGE_END` and keeps only that
interval (plus the final head on the final stage) on the GPU.

This service is an experiment implementation, not the permanent Swarm
topology. The stage split is a temporary controlled variable for testing the
state-local protocol hypothesis.

