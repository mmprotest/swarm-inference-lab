"""Checkpoint-authoritative Kimi K3 tokenizer and chat-wire certification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA_VERSION = "experiment-014-k3-conversation-semantics-v1"

# Copied byte-for-byte from the immutable checkpoint's tokenization_kimi.py.
_PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


class ConversationCertificationError(ValueError):
    """The official renderer or production chat wire is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ConversationCertificationError(f"cannot load Python source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _official_tokenizer(checkpoint: Path) -> tuple[Any, dict[str, int]]:
    try:
        import tiktoken  # type: ignore[import-not-found]
        from tiktoken.load import load_tiktoken_bpe  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConversationCertificationError(
            "the independent tokenizer environment needs pinned tiktoken"
        ) from exc
    config = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    ranks = load_tiktoken_bpe(str(checkpoint / "tiktoken.model"))
    decoder = {
        int(token_id): row["content"]
        for token_id, row in config["added_tokens_decoder"].items()
    }
    base = len(ranks)
    special_tokens = {
        decoder.get(token_id, f"<|reserved_token_{token_id}|>"): token_id
        for token_id in range(base, base + 256)
    }
    encoding = tiktoken.Encoding(
        name="Kimi-K3-checkpoint-reference",
        pat_str=_PATTERN,
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )
    return encoding, special_tokens


def _encode_segments(encoding: Any, segments: list[Any]) -> list[int]:
    result: list[int] = []
    for segment in segments:
        if segment.allow_special:
            result.extend(encoding.encode(segment.text, allowed_special="all"))
        else:
            result.extend(encoding.encode(segment.text, disallowed_special=()))
    return result


def _fixture_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "single_turn_thinking_default_max",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": True,
            "reasoning_effort": "max",
        },
        {
            "name": "multi_turn_reasoning_low",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hello\nKimi"},
                {
                    "role": "assistant",
                    "reasoning_content": "because",
                    "content": "Hello.",
                },
                {"role": "user", "content": "Continue"},
            ],
            "thinking": True,
            "reasoning_effort": "low",
        },
        {
            "name": "multi_turn_no_thinking",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Second"},
            ],
            "thinking": False,
            "reasoning_effort": None,
        },
    ]


def certify_conversation_semantics(
    checkpoint: Path,
    tokenizer_json: Path,
    executable: Path,
    openai_server_source: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    tokenizer_path = tokenizer_json.expanduser().resolve()
    executable_path = executable.expanduser().resolve()
    server_path = openai_server_source.expanduser().resolve()
    for required in (
        root / "encoding_k3.py",
        root / "tokenization_kimi.py",
        root / "tokenizer_config.json",
        root / "tiktoken.model",
        tokenizer_path,
        executable_path,
        server_path,
    ):
        if not required.is_file():
            raise ConversationCertificationError(f"missing required input {required}")
    official = _load_module("k3_checkpoint_encoding", root / "encoding_k3.py")
    server = _load_module("k3_product_openai_server", server_path)
    encoding, specials = _official_tokenizer(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    destination = output_path.expanduser().resolve()
    fixtures_directory = destination.parent / "conversation-fixtures"
    fixtures_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in _fixture_cases():
        official_segments = official.build_chat_segments(
            case["messages"],
            thinking=case["thinking"],
            thinking_effort=case["reasoning_effort"],
        )
        expected = [
            int(config["bos_token_id"]),
            *_encode_segments(encoding, official_segments),
        ]
        wire = server.render_chat_kimi(
            case["messages"],
            enable_thinking=case["thinking"],
            reasoning_effort=case["reasoning_effort"],
        )
        wire_path = fixtures_directory / f"{case['name']}.wire"
        wire_path.write_bytes(wire.encode("utf-8"))
        environment = os.environ.copy()
        environment["K3_TOKENIZER"] = str(tokenizer_path)
        process = subprocess.run(
            [str(executable_path), str(root), "--wire-test", str(wire_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            env=environment,
        )
        try:
            # --wire-test returns the renderer payload only; the production
            # serve_one path prepends the checkpoint BOS immediately before it.
            actual = [int(config["bos_token_id"])] + [
                int(value) for value in process.stdout.split()
            ]
        except ValueError:
            actual = []
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(actual, expected, strict=False))
                if left != right
            ),
            None,
        )
        if mismatch is None and len(actual) != len(expected):
            mismatch = min(len(actual), len(expected))
        rows.append(
            {
                "name": case["name"],
                "status": "PASS"
                if process.returncode == 0 and actual == expected
                else "FAIL",
                "thinking": case["thinking"],
                "reasoning_effort": case["reasoning_effort"],
                "token_count": len(expected),
                "token_equality": actual == expected,
                "first_mismatch_index": mismatch,
                "official_mismatch_window": expected[
                    max(0, (mismatch or 0) - 2) : (mismatch or 0) + 6
                ]
                if mismatch is not None
                else [],
                "product_mismatch_window": actual[
                    max(0, (mismatch or 0) - 2) : (mismatch or 0) + 6
                ]
                if mismatch is not None
                else [],
                "official_token_sha256": hashlib.sha256(
                    json.dumps(expected, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "product_token_sha256": hashlib.sha256(
                    json.dumps(actual, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "wire_sha256": _sha256(wire_path),
                "engine_return_code": process.returncode,
                "engine_stderr": process.stderr.strip(),
            }
        )
    tool_rejection = False
    try:
        server.render_chat_kimi(
            [{"role": "user", "content": "Use a tool"}],
            tools=[{"type": "function", "function": {"name": "fixture"}}],
        )
    except server.APIError:
        tool_rejection = True
    eos_id = int(config["eos_token_id"])
    eos_matches_eom = specials.get("<|end_of_msg|>") == eos_id
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in rows)
        and tool_rejection
        and eos_matches_eom
        else "FAIL"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "checkpoint": str(root),
        "checkpoint_sources": {
            "encoding_k3_sha256": _sha256(root / "encoding_k3.py"),
            "tokenization_kimi_sha256": _sha256(root / "tokenization_kimi.py"),
            "tiktoken_model_sha256": _sha256(root / "tiktoken.model"),
        },
        "product_sources": {
            "openai_server_sha256": _sha256(server_path),
            "executable_sha256": _sha256(executable_path),
            "generated_tokenizer_sha256": _sha256(tokenizer_path),
        },
        "fixtures": rows,
        "eos": {
            "checkpoint_eos_token_id": eos_id,
            "end_of_message_token_id": specials.get("<|end_of_msg|>"),
            "eos_matches_end_of_message": eos_matches_eom,
        },
        "initial_product_tool_calls": {
            "supported": False,
            "unsupported_request_rejected": tool_rejection,
            "policy": "fail closed until the official XTML tool subset is wired end to end",
        },
        "certified_cases": [
            "single-turn prompt",
            "multi-turn history",
            "reasoning-content preservation",
            "thinking effort low/max",
            "non-thinking response channel",
            "EOS/end-of-message identity",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "fixture_count": len(rows),
        "passing_fixtures": sum(row["status"] == "PASS" for row in rows),
        "tool_calls_fail_closed": tool_rejection,
    }


__all__ = [
    "ConversationCertificationError",
    "certify_conversation_semantics",
]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--openai-server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = certify_conversation_semantics(
        args.checkpoint,
        args.tokenizer_json,
        args.executable,
        args.openai_server,
        args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
