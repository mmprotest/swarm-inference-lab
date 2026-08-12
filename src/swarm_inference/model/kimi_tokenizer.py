"""Exact-hash local trust boundary for the Kimi K3 tokenizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from swarm_inference.exceptions import IntegrityError

KIMI_TOKENIZER_ASSETS = (
    "encoding_k3.py",
    "tiktoken.model",
    "tokenization_kimi.py",
    "tokenizer_config.json",
)


def apply_kimi_prompt_special_tokens(
    tokenizer: Any,
    token_ids: list[int],
    *,
    add_special_tokens: bool,
) -> list[int]:
    """Apply the native Kimi engine's explicit, exactly-once BOS rule."""

    normalized = [int(value) for value in token_ids]
    if not add_special_tokens:
        return normalized
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if (
        isinstance(bos_token_id, bool)
        or not isinstance(bos_token_id, int)
        or bos_token_id < 0
    ):
        raise IntegrityError("configured Kimi tokenizer has no valid BOS token ID")
    if not normalized or normalized[0] != bos_token_id:
        normalized.insert(0, bos_token_id)
    return normalized


def verify_pinned_kimi_tokenizer_assets(
    model_path: str | Path,
    identity_path: str | Path,
    *,
    expected_worker_id: str,
) -> dict[str, Any]:
    """Verify the complete tokenizer allowlist and stage-zero ownership."""

    snapshot = Path(model_path).expanduser().resolve()
    identity_file = Path(identity_path).expanduser().resolve()
    try:
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("configured Kimi tokenizer identity is invalid") from exc
    if not isinstance(identity, dict):
        raise IntegrityError("configured Kimi tokenizer identity is not an object")
    if identity.get("schema_version") != "swarm-model-identity-v1":
        raise IntegrityError("configured Kimi tokenizer identity schema is unsupported")
    if identity.get("worker_id") != expected_worker_id:
        raise IntegrityError("configured Kimi tokenizer belongs to a different worker")
    if identity.get("adapter_id") != "kimi_k3_cuda":
        raise IntegrityError("configured tokenizer is not bound to the Kimi CUDA adapter")
    if identity.get("owns_embeddings") is not True:
        raise IntegrityError("Kimi prompt tokenization is restricted to stage zero")
    hashes = identity.get("tokenizer_assets_sha256")
    if not isinstance(hashes, dict) or sorted(hashes) != list(KIMI_TOKENIZER_ASSETS):
        raise IntegrityError("configured Kimi tokenizer allowlist is incomplete or expanded")
    for name in KIMI_TOKENIZER_ASSETS:
        expected = hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise IntegrityError(f"configured Kimi tokenizer hash is invalid: {name}")
        asset = snapshot / name
        if asset.is_symlink() or not asset.is_file():
            raise IntegrityError(f"configured Kimi tokenizer asset is absent: {name}")
        actual = hashlib.sha256(asset.read_bytes()).hexdigest()
        if actual != expected:
            raise IntegrityError(f"configured Kimi tokenizer asset hash differs: {name}")
    return identity


def load_pinned_kimi_tokenizer(
    model_path: str | Path,
    identity_path: str | Path,
    *,
    expected_worker_id: str,
) -> Any:
    """Load Kimi custom tokenizer code only after local allowlist verification."""

    verify_pinned_kimi_tokenizer_assets(
        model_path,
        identity_path,
        expected_worker_id=expected_worker_id,
    )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        Path(model_path).expanduser().resolve(),
        local_files_only=True,
        trust_remote_code=True,
    )


__all__ = [
    "KIMI_TOKENIZER_ASSETS",
    "apply_kimi_prompt_special_tokens",
    "load_pinned_kimi_tokenizer",
    "verify_pinned_kimi_tokenizer_assets",
]
