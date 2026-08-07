"""User-facing model source references.

This module parses identity syntax only.  It deliberately does not inspect a
model, choose a quantisation, download files, or select an execution engine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from pydantic import ConfigDict, Field, model_validator

from swarm_inference.config.models import StrictModel

_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})


class ModelSourceReference(StrictModel):
    """A parsed model source before any mutable reference is resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: Literal["huggingface", "local"]
    model_id: str = Field(min_length=1)
    requested_revision: str | None = None
    variant: str | None = None
    local_path: Path | None = None

    @model_validator(mode="after")
    def validate_source(self) -> ModelSourceReference:
        if self.source_type == "local" and self.local_path is None:
            raise ValueError("local model references require a path")
        if self.source_type == "huggingface" and self.local_path is not None:
            raise ValueError("Hugging Face references cannot include a local path")
        return self

    @classmethod
    def parse(
        cls,
        value: str | Path,
        *,
        revision: str | None = None,
        variant: str | None = None,
    ) -> ModelSourceReference:
        return parse_model_source(value, revision=revision, variant=variant)


def _looks_local(value: str) -> bool:
    path = Path(value).expanduser()
    return (
        bool(_WINDOWS_PATH.match(value))
        or value.startswith(("/", "./", "../", "~", "\\\\"))
        or value.lower().endswith(".gguf")
        or path.exists()
    )


def _parse_hugging_face_url(value: str) -> tuple[str, str | None]:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _HF_HOSTS:
        raise ValueError("only huggingface.co model URLs are supported")
    parts = [unquote(item) for item in parsed.path.split("/") if item]
    if len(parts) < 2:
        raise ValueError("Hugging Face model URLs must contain an owner and repository")
    model_id = "/".join(parts[:2])
    url_revision: str | None = None
    if len(parts) >= 4 and parts[2] in {"tree", "resolve"}:
        url_revision = parts[3]
    return model_id, url_revision


def parse_model_source(
    value: str | Path,
    *,
    revision: str | None = None,
    variant: str | None = None,
) -> ModelSourceReference:
    """Parse supported local, Hub-ID, and Hugging Face URL syntax."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("model reference cannot be empty")
    if _looks_local(raw):
        local = Path(raw).expanduser()
        return ModelSourceReference(
            source_type="local",
            model_id=str(local),
            requested_revision=revision,
            variant=variant,
            local_path=local,
        )
    if raw.startswith(("http://", "https://")):
        model_id, url_revision = _parse_hugging_face_url(raw)
        if revision is not None and url_revision is not None and revision != url_revision:
            raise ValueError("URL revision conflicts with the explicit revision override")
        return ModelSourceReference(
            source_type="huggingface",
            model_id=model_id,
            requested_revision=revision or url_revision,
            variant=variant,
        )

    reference = raw
    parsed_revision: str | None = None
    if "@" in reference:
        reference, parsed_revision = reference.rsplit("@", 1)
        if not parsed_revision:
            raise ValueError("model revision cannot be empty")
    parsed_variant: str | None = None
    slash = reference.find("/")
    colon = reference.rfind(":")
    if slash >= 0 and colon > slash:
        reference, parsed_variant = reference[:colon], reference[colon + 1 :]
        if not parsed_variant:
            raise ValueError("model variant cannot be empty")
    if reference.count("/") != 1 or any(not item for item in reference.split("/")):
        raise ValueError(
            "Hugging Face model references must use owner/repository syntax or a local path"
        )
    if revision is not None and parsed_revision is not None and revision != parsed_revision:
        raise ValueError("model reference revision conflicts with the explicit override")
    if variant is not None and parsed_variant is not None and variant != parsed_variant:
        raise ValueError("model reference variant conflicts with the explicit override")
    return ModelSourceReference(
        source_type="huggingface",
        model_id=reference,
        requested_revision=revision or parsed_revision,
        variant=variant or parsed_variant,
    )


__all__ = ["ModelSourceReference", "parse_model_source"]
