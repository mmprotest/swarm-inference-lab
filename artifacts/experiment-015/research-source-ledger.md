# Experiment 015 research source ledger

External results motivate hypotheses only; none are relabelled as Swarm measurements.

| ID | Topic | Source | Use | Imported into Swarm result |
| --- | --- | --- | --- | --- |
| R015-001 | Kimi K3 serving and DSpark/DCP | [Kimi K3 Is Here: Efficient Day-0 Support on vLLM](https://vllm-project.github.io/2026/07/27/k3.html) | hypothesis motivation and external reference only | NO |
| R015-002 | Kimi K3 DSpark | [Inferact/Kimi-K3-DSpark model card](https://huggingface.co/Inferact/Kimi-K3-DSpark/blob/main/README.md) | checkpoint identity, architecture, public acceptance sensitivity | NO |
| R015-003 | DSpark | [DSpark: Confidence-Scheduled Speculative Decoding](https://arxiv.org/abs/2607.05147) | confidence scheduling and verification-waste hypothesis | NO |
| R015-004 | speculative decoding | [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) | distribution-preserving rejection-sampling semantics | NO |
| R015-005 | tree speculative inference | [SpecInfer](https://arxiv.org/abs/2305.09781) | token-tree branch criterion | NO |
| R015-006 | pipeline speculative inference | [PipeInfer](https://arxiv.org/abs/2407.11798) | asynchronous speculation and early cancellation hypothesis | NO |
| R015-007 | hierarchical speculative pipelines | [PipeSpec](https://aclanthology.org/2025.findings-acl.669/) | hierarchical pipeline hypothesis | NO |
| R015-008 | expert parallelism | [DeepEP](https://github.com/deepseek-ai/DeepEP) | device-resident dispatch/combine and overlap principles | NO |
| R015-009 | expert load balancing | [Expert Parallelism Load Balancer](https://github.com/deepseek-ai/EPLB) | redundant expert placement hypothesis | NO |
| R015-010 | Decode Context Parallelism | [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) | DCP exact-combine and conditional-admission design | NO |
| R015-011 | on-demand expert loading | [OD-MoE](https://arxiv.org/abs/2512.03927) | tiered residency hypothesis | NO |
| R015-012 | hybrid expert offload | [HybriMoE](https://arxiv.org/abs/2504.05897) | dynamic CPU/GPU scheduling, prefetch, and caching hypothesis | NO |
| R015-013 | communication/computation overlap | [DeepSeek public profiling data](https://github.com/deepseek-ai/profile-data) | operation-timeline/nanobatching hypothesis | NO |
