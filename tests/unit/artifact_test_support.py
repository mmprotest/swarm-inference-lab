from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from swarm_inference.model.adapter import partition_metadata_from_description
from swarm_inference.model.partition import StagePlan, build_stage_plan
from swarm_inference.model.shard_builder import ResolvedModel, inspect_native_model

MODEL_ID = "test/tiny-native-model"
MODEL_REVISION = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"


def tiny_native_snapshot(tmp_path: Path, *, stage_count: int = 4) -> tuple[Path, str, StagePlan]:
    torch.manual_seed(8675309)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            tie_word_embeddings=False,
            attention_dropout=0,
            pad_token_id=0,
            eos_token_id=31,
        )
    ).eval()
    snapshot = tmp_path / "source"
    model.save_pretrained(snapshot, safe_serialization=True, max_shard_size="2KB")
    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_commit_hash"] = MODEL_REVISION
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    tokenizer_payload = b'{"version":"1.0","test":"tiny-native-model"}'
    (snapshot / "tokenizer.json").write_bytes(tokenizer_payload)
    (snapshot / "tokenizer_config.json").write_text('{"model_max_length":32}', encoding="utf-8")
    tokenizer_revision = "sha256:" + hashlib.sha256(tokenizer_payload).hexdigest()
    description = inspect_native_model(
        ResolvedModel(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            path=snapshot,
            downloaded=False,
        )
    )
    metadata = partition_metadata_from_description(
        description,
        tokenizer_revision=tokenizer_revision,
    )
    plan = build_stage_plan(
        snapshot,
        metadata=metadata,
        stage_count=stage_count,
        method="equal",
        memory_limit_bytes=1024**3,
        device="cpu",
    )
    return snapshot, tokenizer_revision, plan
