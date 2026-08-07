"""Architecture-neutral checkpoint quantization metadata inspection."""

from __future__ import annotations

import re
from typing import Any


def normalize_quantization_name(value: str | None) -> str | None:
    """Normalize equivalent format spellings without guessing from model IDs."""

    if value is None or not value.strip():
        return None
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if "mxfp4" in normalized:
        return "mxfp4"
    if "fp8" in normalized and "e4m3" in normalized:
        return "fp8-e4m3"
    aliases = {
        "bfloat16": "bf16",
        "float16": "f16",
        "float32": "f32",
    }
    return aliases.get(normalized, normalized)


def _weight_schemes(raw: dict[str, Any]) -> tuple[str, ...]:
    groups = raw.get("config_groups")
    if not isinstance(groups, dict):
        return ()
    schemes: list[str] = []
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if not isinstance(weights, dict):
            continue
        bits = weights.get("num_bits", weights.get("bits"))
        kind = str(weights.get("type", "")).casefold()
        strategy = str(weights.get("strategy", "")).casefold()
        group_size = weights.get("group_size")
        group_format = normalize_quantization_name(
            str(group.get("format")) if group.get("format") is not None else None
        )
        rendered = " ".join(str(value) for value in (*group.values(), *weights.values()))
        if group_format == "mxfp4" or "mxfp4" in rendered.casefold():
            schemes.append("mxfp4")
            continue
        if isinstance(bits, int) and bits > 0 and kind in {"int", "integer"}:
            scheme = f"int{bits}"
            if strategy == "group" and isinstance(group_size, int) and group_size > 0:
                scheme += f"-g{group_size}"
            schemes.append(scheme)
    return tuple(dict.fromkeys(schemes))


def quantization_from_config(config: dict[str, Any]) -> str | None:
    """Derive the numerical storage scheme from explicit configuration metadata."""

    nested = config.get("text_config")
    owners = (config, nested) if isinstance(nested, dict) else (config,)
    discovered: list[str] = []
    for owner in owners:
        raw = owner.get("quantization_config")
        if not isinstance(raw, dict):
            continue
        rendered = " ".join(str(value) for value in raw.values()).casefold()
        if "mxfp4" in rendered:
            discovered.append("mxfp4")
            continue
        schemes = _weight_schemes(raw)
        if schemes:
            discovered.extend(schemes)
            continue
        method = normalize_quantization_name(
            str(raw.get("quant_method") or raw.get("quantization_method") or "")
        )
        storage_format = normalize_quantization_name(
            str(raw.get("format") or raw.get("fmt") or raw.get("format_name") or "")
        )
        bits = raw.get("bits", raw.get("num_bits"))
        if method == "fp8" and storage_format and "e4m3" in storage_format:
            discovered.append("fp8-e4m3")
        elif method == "fp8":
            discovered.append("fp8")
        elif isinstance(bits, int) and bits > 0:
            discovered.append(f"{method + '-' if method else ''}int{bits}")
        elif method and method != "compressed-tensors":
            discovered.append(method)
        elif storage_format and storage_format not in {"pack-quantized", "compressed"}:
            discovered.append(storage_format)
    values = tuple(dict.fromkeys(discovered))
    if not values:
        return None
    return values[0] if len(values) == 1 else "mixed:" + "+".join(values)


__all__ = ["normalize_quantization_name", "quantization_from_config"]
