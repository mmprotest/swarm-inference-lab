"""Exact PyTorch stage-rank adapter for canonical safetensors/microshards."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from swarm_inference.config.models import (
    AttentionConfig,
    Backend,
    CacheSpec,
    ModelManifest,
    OperationKind,
    StageDefinition,
    TensorSpec,
)
from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.qwen3 import Qwen3StageModule
from swarm_inference.model.qwen3_runtime import Qwen3EngineOptions, Qwen3ExecutionProfile
from swarm_inference.worker.abi import (
    BackendAdapter,
    ResultClassification,
    TensorPayload,
    TokenPayload,
    WorkerBenchmarkProfile,
    WorkerCapabilities,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
    tensor_payload_from_array,
)


class TorchRankAdapter(BackendAdapter):
    """One real stage in one process; no whole-model or synthetic fallback."""

    supported_jobs = frozenset(
        {
            WorkerJobType.PIPELINE_STAGE_PREFILL,
            WorkerJobType.PIPELINE_STAGE_DECODE,
            WorkerJobType.TENSOR_RANK,
            WorkerJobType.INTEGRITY_AUDIT,
        }
    )

    def __init__(
        self,
        *,
        module: Qwen3StageModule,
        capabilities: WorkerCapabilities,
        model_manifest: ModelManifest,
        partition_hash: str,
        shard_hash: str,
        device_type: str,
        model_load_seconds: float,
        warmup_seconds: float,
    ) -> None:
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("Torch rank device must be cpu or cuda")
        self.module = module
        self._capabilities = capabilities
        self.model_manifest = model_manifest
        self.partition_hash = partition_hash
        self.shard_hash = shard_hash
        self.device_type = device_type
        self.backend_id = f"torch-{device_type}"
        self._profile = WorkerBenchmarkProfile(
            model_revision=model_manifest.model_revision,
            shard_hash=shard_hash,
            model_load_seconds=model_load_seconds,
            warmup_seconds=warmup_seconds,
        )
        self._cancelled: set[str] = set()
        self._execution_samples_ms: list[float] = []

    @classmethod
    def from_microshard(
        cls,
        *,
        partition_root: Path,
        stage_id: int,
        device: str,
        capabilities: WorkerCapabilities,
        partition_hash: str,
        dtype: str = "bfloat16",
        warmup_sequence_length: int = 8,
    ) -> TorchRankAdapter:
        import torch

        root = partition_root.expanduser().resolve()
        manifest, shard_directory, shard_hash = _model_manifest_from_microshards(root, stage_id)
        config = json.loads((root / "config" / "config.json").read_text(encoding="utf-8"))
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype)
        if torch_dtype is None:
            raise IntegrityError(f"unsupported torch rank dtype {dtype!r}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA rank requested but CUDA is unavailable")
        stage = manifest.stages[stage_id]
        load_started = time.perf_counter()
        module = Qwen3StageModule(
            config=config,
            stage=stage,
            device=device,
            dtype=torch_dtype,
            engine_options=Qwen3EngineOptions.from_values(
                profile=Qwen3ExecutionProfile.CORRECTNESS,
                max_sequence_length=min(4096, int(config["max_position_embeddings"])),
                boundary_diagnostics=True,
            ),
        )
        module.load_weights(shard_directory, manifest=manifest)
        load_seconds = time.perf_counter() - load_started
        warm_started = time.perf_counter()
        module.warmup(sequence_length=warmup_sequence_length)
        warmup_seconds = time.perf_counter() - warm_started
        return cls(
            module=module,
            capabilities=capabilities,
            model_manifest=manifest,
            partition_hash=partition_hash,
            shard_hash=shard_hash,
            device_type=device,
            model_load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,
        )

    def capabilities(self) -> WorkerCapabilities:
        return self._capabilities

    def benchmark_profile(self) -> WorkerBenchmarkProfile:
        return self._profile

    def admission_result(self, job: WorkerJob) -> WorkerJobResult | None:
        rejected = super().admission_result(job)
        if rejected is not None:
            return rejected
        if self.module.required_memory_bytes > self._capabilities.maximum_weight_bytes:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.INSUFFICIENT_MEMORY,
                detail=(
                    f"stage requires {self.module.required_memory_bytes} bytes, worker admits "
                    f"{self._capabilities.maximum_weight_bytes}"
                ),
            )
        if self._execution_samples_ms:
            estimate = float(np.percentile(self._execution_samples_ms, 95))
            if estimate > job.remaining_deadline_ms:
                return WorkerJobResult(
                    job_id=job.job_id,
                    request_id=job.request_id,
                    status=WorkerJobStatus.DEADLINE_IMPOSSIBLE,
                    detail=f"measured p95 stage time {estimate:.3f} ms exceeds remaining deadline",
                )
        return None

    async def execute(self, job: WorkerJob) -> WorkerJobResult:
        rejected = self.admission_result(job)
        if rejected is not None:
            return rejected
        if job.model_revision != self.model_manifest.model_revision:
            return self._unsupported(job, "model revision mismatch")
        if job.partition_manifest_hash != self.partition_hash:
            return self._unsupported(job, "partition manifest hash mismatch")
        if job.shard_hash != self.shard_hash:
            return self._unsupported(job, "stage shard hash mismatch")
        try:
            activation, logical_dtype, token_position, sequence_length = self._activation(job)
        except (IntegrityError, ValueError) as exc:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.INCOMPATIBLE_DTYPE,
                detail=str(exc),
            )
        operation = (
            OperationKind.PREFILL
            if job.role == WorkerJobType.PIPELINE_STAGE_PREFILL
            else OperationKind.DECODE
        )
        if job.role in {WorkerJobType.TENSOR_RANK, WorkerJobType.INTEGRITY_AUDIT}:
            operation = OperationKind(str(job.metadata.get("operation", "prefill")))
        started = time.perf_counter()
        try:
            output = await asyncio.to_thread(
                self.module.execute,
                activation,
                request_id=job.request_id,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                cache_generation=int(job.metadata.get("cache_generation", 0)),
                route_generation=job.route_generation,
            )
        except Exception as exc:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.BACKEND_FAILURE,
                detail=f"PyTorch rank execution failed: {type(exc).__name__}: {exc}",
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._execution_samples_ms.append(elapsed_ms)
        if job.request_id in self._cancelled:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.CANCELLED,
            )
        output_array = np.asarray(output)
        output_logical_dtype = "bfloat16" if output_array.dtype == np.uint16 else None
        token_count = max(sequence_length, 1)
        rate = token_count / max(elapsed_ms / 1000, 1e-12)
        if operation == OperationKind.PREFILL:
            self._profile.prefill_tokens_per_second = rate
        else:
            self._profile.decode_tokens_per_second = rate
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=WorkerJobStatus.ACCEPTED,
            output_payload=tensor_payload_from_array(
                output_array,
                tensor_id=f"{job.job_id}-stage-{self.module.stage_id}-output",
                request_id=job.request_id,
                stage_id=self.module.stage_id,
                token_position=token_position,
                sequence_length=int(output_array.shape[1]),
                model_revision=self.model_manifest.model_revision,
                partition_hash=self.partition_hash,
                route_generation=job.route_generation,
                logical_dtype=output_logical_dtype,
            ),
            metrics={
                "execution_ms": elapsed_ms,
                "cache_bytes": self.module.cache_bytes(),
                "stage_id": self.module.stage_id,
                "stage_offset": self.module.stage.layer_start,
                "device": self.device_type,
                "input_logical_dtype": logical_dtype,
                "output_logical_dtype": output_logical_dtype or str(output_array.dtype),
                "synthetic_fallback": False,
                "state": self.module.state_summary(),
            },
            classification=(
                ResultClassification.MEASURED_CUDA
                if self.device_type == "cuda"
                else ResultClassification.MEASURED_X86_CPU
            ),
        )

    def _activation(self, job: WorkerJob) -> tuple[np.ndarray, str, int, int]:
        if self.module.stage.owns_embeddings:
            if not isinstance(job.input_payload, TokenPayload):
                raise ValueError("embedding stage requires token IDs")
            values = np.asarray(job.input_payload.token_ids, dtype=np.int64)[None, :]
            return values, "int64", int(job.metadata.get("token_position", 0)), values.shape[1]
        if not isinstance(job.input_payload, TensorPayload):
            raise ValueError("non-embedding stage requires a tensor payload")
        tensor = job.input_payload.to_tensor()
        if tensor.model_revision != self.model_manifest.model_revision:
            raise IntegrityError("activation model revision mismatch")
        if tensor.partition_hash != self.partition_hash:
            raise IntegrityError("activation partition hash mismatch")
        if tensor.route_generation != job.route_generation:
            raise IntegrityError("activation route generation mismatch")
        logical_dtype = tensor.logical_dtype or str(tensor.array.dtype)
        if logical_dtype not in self._capabilities.supported_activation_dtypes:
            raise ValueError(f"worker does not support activation dtype {logical_dtype}")
        return tensor.array, logical_dtype, tensor.token_position, tensor.sequence_length

    def _unsupported(self, job: WorkerJob, detail: str) -> WorkerJobResult:
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=WorkerJobStatus.UNSUPPORTED,
            detail=detail,
        )

    async def cancel(self, request_id: str) -> bool:
        self._cancelled.add(request_id)
        self.module.cancel(request_id)
        return True


def _model_manifest_from_microshards(
    root: Path,
    selected_stage_id: int,
) -> tuple[ModelManifest, Path, str]:
    partition = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "parallel_plan.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config" / "config.json").read_text(encoding="utf-8"))
    if int(partition["tensor_parallel_degree"]) != 1:
        raise IntegrityError("Torch stage adapter requires a TP1 microshard")
    stages: list[StageDefinition] = []
    shard_hashes: dict[str, str] = {}
    per_layer_bytes = [0 for _ in range(int(config["num_hidden_layers"]))]
    embedding_bytes = 0
    output_head_bytes = 0
    for raw_stage in plan["pipeline_stages"]:
        stage_id = int(raw_stage["stage_id"])
        layer_ids = [int(item["layer_id"]) for item in raw_stage["layer_plans"]]
        if not layer_ids:
            raise IntegrityError(f"stage {stage_id} has no layers")
        shard_directory = root / "stages" / f"stage-{stage_id:03d}" / "ranks" / "rank-000"
        shard_manifest = json.loads(
            (shard_directory / "shard_manifest.json").read_text(encoding="utf-8")
        )
        tensors = shard_manifest["tensors"]
        tensor_names = [str(item["tensor_name"]) for item in tensors]
        for item in tensors:
            name = str(item["tensor_name"])
            size = int(item["logical_bytes"])
            if name.startswith("model.layers."):
                layer_id = int(name.split(".")[2])
                per_layer_bytes[layer_id] += size
            elif name.startswith("model.embed_tokens"):
                embedding_bytes += size
            elif name.startswith("lm_head"):
                output_head_bytes += size
        required = int(shard_manifest["logical_weight_bytes"])
        hidden_size = int(config["hidden_size"])
        input_dtype = "int64" if bool(raw_stage["owns_embeddings"]) else "bfloat16"
        stages.append(
            StageDefinition(
                stage_id=stage_id,
                layer_start=min(layer_ids),
                layer_end=max(layer_ids) + 1,
                owns_embeddings=bool(raw_stage["owns_embeddings"]),
                owns_final_norm=bool(raw_stage["owns_final_norm"]),
                owns_output_head=bool(raw_stage["owns_lm_head"]),
                required_memory_bytes=required,
                required_total_memory_bytes=required,
                input_spec=TensorSpec(
                    dtype=input_dtype,
                    shape=["batch", "sequence", hidden_size],
                ),
                output_spec=TensorSpec(
                    dtype=("float32" if bool(raw_stage["owns_lm_head"]) else "bfloat16"),
                    shape=["batch", "sequence", hidden_size],
                ),
                cache_spec=CacheSpec(
                    format="dynamic-kv",
                    bytes_per_token=(
                        len(layer_ids)
                        * int(config["num_key_value_heads"])
                        * int(
                            config.get(
                                "head_dim", hidden_size // int(config["num_attention_heads"])
                            )
                        )
                        * 2
                        * 2
                    ),
                ),
                tensor_names=tensor_names,
                tensor_count=len(tensor_names),
                shard_hash=str(shard_manifest["weight_file_hash"]),
            )
        )
        shard_hashes[str(stage_id)] = str(shard_manifest["weight_file_hash"])
    layer_count = int(config["num_hidden_layers"])
    total_weight_bytes = int(partition["source_weight_bytes"])
    manifest = ModelManifest(
        schema_version="universal-worker-abi-v1",
        model_id=str(partition["model_id"]),
        model_revision=str(partition["model_revision"]),
        architecture="Qwen3ForCausalLM",
        tokenizer_id=str(partition["model_id"]),
        layer_count=layer_count,
        hidden_size=int(config["hidden_size"]),
        attention=AttentionConfig(
            head_count=int(config["num_attention_heads"]),
            key_value_head_count=int(config["num_key_value_heads"]),
            head_dimension=int(
                config.get(
                    "head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"])
                )
            ),
            rope_theta=float(config.get("rope_theta", 1_000_000.0)),
        ),
        vocabulary_size=int(config["vocab_size"]),
        weight_dtype="BF16",
        total_weight_bytes=total_weight_bytes,
        embedding_bytes=embedding_bytes,
        output_head_bytes=output_head_bytes,
        per_layer_weight_bytes=per_layer_bytes,
        estimated_cache_bytes_per_token_per_layer=(
            int(config["num_key_value_heads"])
            * int(
                config.get(
                    "head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"])
                )
            )
            * 2
            * 2
        ),
        activation_bytes_per_stage_boundary=int(config["hidden_size"]) * 2,
        stages=stages,
        shard_hashes=shard_hashes,
        compatible_worker_backends=[Backend.TORCH_CPU, Backend.TORCH_CUDA],
        source_tensor_hashes={},
        source_files=dict(partition.get("source_file_hashes", {})),
        final_normalisation_owner=len(stages) - 1,
        lm_head_owner=len(stages) - 1,
        supported_dtypes=["bfloat16", "float32"],
    )
    if not 0 <= selected_stage_id < len(stages):
        raise IntegrityError(f"stage {selected_stage_id} is outside the partition")
    selected_directory = root / "stages" / f"stage-{selected_stage_id:03d}" / "ranks" / "rank-000"
    return manifest, selected_directory, shard_hashes[str(selected_stage_id)]
