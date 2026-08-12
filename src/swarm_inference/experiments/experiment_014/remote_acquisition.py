"""Immutable, resumable, worker-scoped Kimi tensor acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

from swarm_inference.exceptions import IntegrityError
from swarm_inference.experiments.experiment_014.distribution import (
    materialize_worker_package,
)
from swarm_inference.model.kimi_tokenizer import (
    KIMI_TOKENIZER_ASSETS,
    apply_kimi_prompt_special_tokens,
    load_pinned_kimi_tokenizer,
    verify_pinned_kimi_tokenizer_assets,
)

SCHEMA_VERSION = "experiment-014-k3-remote-acquisition-v1"
FIXTURE_SCHEMA_VERSION = "experiment-014-k3-remote-acquisition-fixture-v1"
SNAPSHOT_SCHEMA_VERSION = "experiment-015-k3-worker-snapshot-v1"
_HEADER_LENGTH = struct.Struct("<Q")


class AcquisitionError(RuntimeError):
    """A remote object or worker activation failed closed."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AcquisitionError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _object_path(cache_dir: Path, sha256: str) -> Path:
    if len(sha256) != 64:
        raise AcquisitionError("immutable object SHA-256 must contain 64 hex characters")
    return cache_dir.resolve() / "objects" / sha256


def _download_object(
    url: str,
    expected_sha256: str,
    cache_dir: Path,
    *,
    retries: int = 4,
    timeout_seconds: float = 30.0,
    backoff_seconds: float = 0.05,
) -> dict[str, Any]:
    if retries < 1:
        raise ValueError("retries must be positive")
    destination = _object_path(cache_dir, expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return {
            "status": "PASS",
            "object_path": str(destination),
            "sha256": expected_sha256,
            "cache_hit": True,
            "downloaded_bytes": 0,
            "attempts": 0,
            "range_resume_used": False,
            "corrupt_attempts_rejected": 0,
        }
    partial = destination.with_suffix(".partial")
    downloaded_bytes = 0
    range_resume_used = False
    corrupt_attempts = 0
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            range_resume_used = True
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", response.getcode()))
                if offset and status != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        downloaded_bytes += len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            observed = _sha256(partial)
            if observed != expected_sha256:
                corrupt_attempts += 1
                errors.append(
                    f"attempt {attempt}: SHA-256 {observed} != {expected_sha256}"
                )
                partial.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(backoff_seconds * attempt)
                continue
            partial.replace(destination)
            return {
                "status": "PASS",
                "object_path": str(destination),
                "sha256": observed,
                "cache_hit": False,
                "downloaded_bytes": downloaded_bytes,
                "attempts": attempt,
                "range_resume_used": range_resume_used,
                "corrupt_attempts_rejected": corrupt_attempts,
            }
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    partial.unlink(missing_ok=True)
    raise AcquisitionError(
        f"immutable object acquisition failed after {retries} attempts: {'; '.join(errors)}"
    )


def _activate_source_view(
    worker_id: str,
    cache_dir: Path,
    objects: list[tuple[str, Path]],
) -> Path:
    view = cache_dir.resolve() / "views" / worker_id
    view.mkdir(parents=True, exist_ok=True)
    for name, source in objects:
        target = view / name
        temporary = target.with_suffix(target.suffix + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(target)
    return view


def _resolve_distribution_placement(
    distribution_path: Path,
    distribution: dict[str, Any],
) -> Path:
    reference = Path(str(distribution.get("placement_manifest", "")))
    if (
        not reference.name
        or reference.is_absolute()
        or reference != Path(reference.name)
        or reference.name in {".", ".."}
    ):
        raise AcquisitionError(
            "distribution placement reference must be one sibling filename"
        )
    placement = (distribution_path.resolve().parent / reference).resolve()
    if placement.parent != distribution_path.resolve().parent:
        raise AcquisitionError("distribution placement reference escapes its package")
    if not placement.is_file() or placement.is_symlink():
        raise AcquisitionError("distribution placement manifest is absent or linked")
    expected = str(distribution.get("placement_manifest_sha256", ""))
    if len(expected) != 64 or _sha256(placement) != expected:
        raise AcquisitionError("distribution placement manifest SHA-256 differs")
    return placement


def acquire_worker_package(
    distribution_manifest_path: Path,
    worker_id: str,
    cache_dir: Path,
    output_path: Path,
    *,
    retries: int = 4,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Download only one worker's immutable source shards and atomically activate it."""
    distribution_path = distribution_manifest_path.resolve()
    distribution = _read(distribution_path)
    if distribution.get("status") != "PASS":
        raise AcquisitionError("distribution manifest is not passing")
    placement_path = _resolve_distribution_placement(distribution_path, distribution)
    worker = next(
        (row for row in distribution["workers"] if row["worker_id"] == worker_id),
        None,
    )
    if worker is None:
        raise AcquisitionError(f"distribution has no worker {worker_id!r}")
    source_hashes = {
        str(name): str(row["sha256"])
        for name, row in distribution["source"]["shards"].items()
    }
    downloads: list[dict[str, Any]] = []
    objects: list[tuple[str, Path]] = []
    for shard in worker["source_shards"]:
        name = str(shard["name"])
        expected_sha = str(shard["sha256"])
        if source_hashes.get(name) != expected_sha:
            raise AcquisitionError(f"worker/source SHA mismatch for {name}")
        result = _download_object(
            str(shard["source_url"]),
            expected_sha,
            cache_dir,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        result["source_name"] = name
        result["source_url"] = str(shard["source_url"])
        downloads.append(result)
        objects.append((name, Path(result["object_path"])))
    view = _activate_source_view(worker_id, cache_dir, objects)
    package = materialize_worker_package(
        placement_path,
        worker_id,
        view,
        output_path,
        verify_source_hashes=source_hashes,
    )
    if output_path.with_suffix(output_path.suffix + ".partial").exists():
        raise AcquisitionError("partial package remained after activation")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "worker_id": worker_id,
        "distribution_manifest": str(distribution_path),
        "distribution_manifest_sha256": _sha256(distribution_path),
        "immutable_revision": distribution["source"]["revision"],
        "source_shard_count": len(downloads),
        "downloaded_bytes": sum(int(row["downloaded_bytes"]) for row in downloads),
        "all_cache_hits": all(bool(row["cache_hit"]) for row in downloads),
        "downloads": downloads,
        "package": package,
        "targeted_acquisition": len(downloads)
        < int(distribution["checkpoint"]["shard_count"]),
        "atomic_activation": True,
        "partial_package_absent": True,
    }


def _package_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        encoded_length = handle.read(_HEADER_LENGTH.size)
        if len(encoded_length) != _HEADER_LENGTH.size:
            raise AcquisitionError("worker package has no Safetensors header length")
        header_length = _HEADER_LENGTH.unpack(encoded_length)[0]
        if not 2 <= header_length <= 1024**3:
            raise AcquisitionError("worker package header length is invalid")
        payload = handle.read(header_length)
        if len(payload) != header_length:
            raise AcquisitionError("worker package header is truncated")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise AcquisitionError("worker package header is not an object")
    return value


def activate_worker_snapshot(
    placement_path: Path,
    worker_id: str,
    package_path: Path,
    config_path: Path,
    output_directory: Path,
    *,
    model_id: str = "moonshotai/Kimi-K3",
    tokenizer_revision: str | None = None,
    adapter_id: str = "kimi_k3_cuda",
) -> dict[str, Any]:
    """Atomically create a directly loadable worker-only Kimi snapshot."""
    placement_source = placement_path.resolve()
    placement = _read(placement_source)
    if placement.get("status") != "PASS":
        raise AcquisitionError("placement manifest is not passing")
    worker = next(
        (row for row in placement["workers"] if row["worker_id"] == worker_id),
        None,
    )
    if worker is None:
        raise AcquisitionError(f"placement has no worker {worker_id!r}")
    package_source = package_path.resolve()
    config_source = config_path.resolve()
    if not package_source.is_file() or not config_source.is_file():
        raise AcquisitionError("worker package or Kimi config is absent")
    expected_config_sha = str(placement["checkpoint"]["config_sha256"])
    if _sha256(config_source) != expected_config_sha:
        raise AcquisitionError("Kimi config hash differs from the placement")
    header = _package_header(package_source)
    metadata = header.get("__metadata__", {})
    expected_names = sorted(
        str(tensor["name"])
        for unit in worker["assignment_units"]
        for tensor in unit["tensors"]
    )
    observed_names = sorted(name for name in header if name != "__metadata__")
    if observed_names != expected_names:
        raise AcquisitionError("worker package tensor names differ from its assignment")
    if (
        str(metadata.get("worker_id")) != worker_id
        or str(metadata.get("checkpoint_fingerprint"))
        != str(placement["checkpoint"]["checkpoint_fingerprint"])
    ):
        raise AcquisitionError("worker package identity differs from its assignment")
    destination = output_directory.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AcquisitionError(f"refusing to replace activated snapshot: {destination}")
    with TemporaryDirectory(
        prefix=f"{destination.name}.partial-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        weights = temporary / "model.safetensors"
        try:
            os.link(package_source, weights)
        except OSError:
            shutil.copyfile(package_source, weights)
        shutil.copyfile(config_source, temporary / "config.json")
        index = {
            "metadata": {
                "checkpoint_fingerprint": placement["checkpoint"][
                    "checkpoint_fingerprint"
                ],
                "total_size": sum(
                    int(tensor["physical_bytes"])
                    for unit in worker["assignment_units"]
                    for tensor in unit["tensors"]
                ),
                "worker_id": worker_id,
            },
            "weight_map": {name: "model.safetensors" for name in expected_names},
        }
        _atomic_json(temporary / "model.safetensors.index.json", index)
        resolved_tokenizer_revision = tokenizer_revision or str(
            placement["checkpoint"]["revision"]
        )
        owns_embeddings = bool(
            "embedding" in worker.get("owned_components", [])
            or worker.get("worker_role") == "embedding_dense_stage"
        )
        tokenizer_assets: dict[str, str] = {}
        if owns_embeddings:
            for name in KIMI_TOKENIZER_ASSETS:
                source = config_source.parent / name
                if not source.is_file():
                    raise AcquisitionError(
                        f"stage-zero tokenizer asset is absent beside config.json: {name}"
                    )
                destination_asset = temporary / name
                shutil.copyfile(source, destination_asset)
                tokenizer_assets[name] = _sha256(destination_asset)
        model_identity = {
            "schema_version": "swarm-model-identity-v1",
            "model_id": model_id,
            "model_revision": placement["checkpoint"]["revision"],
            "tokenizer_revision": resolved_tokenizer_revision,
            "adapter_id": adapter_id,
            "model_content_fingerprint": placement["checkpoint"][
                "checkpoint_fingerprint"
            ],
            "config_sha256": _sha256(temporary / "config.json"),
            "safetensors_index_sha256": _sha256(
                temporary / "model.safetensors.index.json"
            ),
            "worker_id": worker_id,
            "assignment_sha256": worker["assignment_sha256"],
            "owns_embeddings": owns_embeddings,
            "tokenizer_assets_sha256": tokenizer_assets,
        }
        _atomic_json(temporary / "model-identity.json", model_identity)
        activation = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "status": "PASS",
            "worker_id": worker_id,
            "checkpoint_fingerprint": placement["checkpoint"][
                "checkpoint_fingerprint"
            ],
            "checkpoint_revision": placement["checkpoint"]["revision"],
            "placement_manifest_sha256": _sha256(placement_source),
            "package_sha256": _sha256(weights),
            "config_sha256": _sha256(temporary / "config.json"),
            "index_sha256": _sha256(temporary / "model.safetensors.index.json"),
            "model_identity_sha256": _sha256(temporary / "model-identity.json"),
            "tokenizer_asset_count": len(tokenizer_assets),
            "tokenizer_asset_bytes": sum(
                (temporary / name).stat().st_size for name in tokenizer_assets
            ),
            "tokenizer_assets_sha256": tokenizer_assets,
            "tensor_count": len(expected_names),
            "source_weight_bytes": int(worker["source_weight_bytes"]),
            "owned_layers": worker["owned_layers"],
            "worker_role": worker["worker_role"],
            "atomic_activation": True,
        }
        _atomic_json(temporary / "activation.json", activation)
        temporary.replace(destination)
    return {
        **activation,
        "snapshot_path": str(destination),
        "activation_receipt_sha256": _sha256(destination / "activation.json"),
        "model_identity_path": str(destination / "model-identity.json"),
        "partial_directory_absent": not any(
            destination.parent.glob(f"{destination.name}.partial-*")
        ),
    }


class _FixtureHandler(BaseHTTPRequestHandler):
    payloads: ClassVar[dict[str, bytes]] = {}
    fail_once: ClassVar[set[str]] = set()
    counts: ClassVar[dict[str, int]] = {}
    ranges: ClassVar[list[str | None]] = []

    def do_GET(self) -> None:
        type(self).counts[self.path] = type(self).counts.get(self.path, 0) + 1
        type(self).ranges.append(self.headers.get("Range"))
        if self.path in type(self).fail_once:
            type(self).fail_once.remove(self.path)
            self.send_response(503)
            self.end_headers()
            return
        payload = type(self).payloads.get(self.path)
        if payload is None:
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        range_header = self.headers.get("Range")
        if range_header:
            prefix = "bytes="
            if not range_header.startswith(prefix) or not range_header.endswith("-"):
                self.send_response(416)
                self.end_headers()
                return
            start = int(range_header[len(prefix) : -1])
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            )
        else:
            self.send_response(200)
        body = payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, message_format: str, *args: object) -> None:
        del message_format, args


def benchmark_remote_acquisition_fixture(
    output_path: Path,
    *,
    cycle_id: str = "H014-037a",
) -> dict[str, Any]:
    """Retain resume/retry/corruption/cache/atomic evidence on a bounded local fixture."""
    payload = bytes(range(256)) * 4096
    corrupt_payload = b"x" * len(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()
    _FixtureHandler.payloads = {"/good": payload, "/corrupt": corrupt_payload}
    _FixtureHandler.fail_once = {"/good"}
    _FixtureHandler.counts = {}
    _FixtureHandler.ranges = []
    receipt: dict[str, Any] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": (
            "Worker-scoped immutable acquisition resumes a partial after one transient "
            "failure, rejects corruption, reuses a verified cache with zero download, "
            "and atomically activates an exact worker Safetensors package."
        ),
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with TemporaryDirectory(prefix="h014-037a-") as temporary_name:
            root = Path(temporary_name)
            cache = root / "cache"
            object_path = _object_path(cache, expected_sha)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            partial = object_path.with_suffix(".partial")
            partial.write_bytes(payload[:65536])
            base = f"http://127.0.0.1:{server.server_port}"
            first = _download_object(
                f"{base}/good", expected_sha, cache, retries=4, timeout_seconds=5.0
            )
            request_count_after_first = sum(_FixtureHandler.counts.values())
            second = _download_object(
                f"{base}/good", expected_sha, cache, retries=4, timeout_seconds=5.0
            )
            request_count_after_second = sum(_FixtureHandler.counts.values())
            corrupt_rejected = False
            corrupt_message = ""
            corrupt_cache = root / "corrupt-cache"
            try:
                _download_object(
                    f"{base}/corrupt",
                    expected_sha,
                    corrupt_cache,
                    retries=2,
                    timeout_seconds=5.0,
                    backoff_seconds=0.0,
                )
            except AcquisitionError as exc:
                corrupt_rejected = True
                corrupt_message = str(exc)

            source_name = "fixture-source.safetensors"
            placement_path = root / "placement.json"
            placement = {
                "status": "PASS",
                "checkpoint": {
                    "revision": "fixture-immutable-revision",
                    "checkpoint_fingerprint": "fixture-checkpoint",
                    "shard_count": 2,
                    "config_sha256": hashlib.sha256(
                        b'{"model_type":"fixture"}\n'
                    ).hexdigest(),
                },
                "workers": [
                    {
                        "worker_id": "fixture-worker",
                        "assignment_sha256": "a" * 64,
                        "checkpoint_fingerprint": "fixture-checkpoint",
                        "worker_role": "whole_layer_stage",
                        "owned_layers": [1],
                        "source_weight_bytes": 384,
                        "assignment_units": [
                            {
                                "unit_id": "fixture-unit",
                                "tensors": [
                                    {
                                        "name": "fixture.tensor",
                                        "safetensors_file": source_name,
                                        "byte_range": [128, 512],
                                        "physical_bytes": 384,
                                        "dtype": "U8",
                                        "shape": [384],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            _atomic_json(placement_path, placement)
            distribution_path = root / "distribution.json"
            distribution = {
                "status": "PASS",
                "checkpoint": placement["checkpoint"],
                "placement_manifest": placement_path.name,
                "placement_manifest_sha256": _sha256(placement_path),
                "source": {
                    "revision": "fixture-immutable-revision",
                    "shards": {
                        source_name: {
                            "sha256": expected_sha,
                            "source_url": f"{base}/good",
                        }
                    },
                },
                "workers": [
                    {
                        "worker_id": "fixture-worker",
                        "source_shards": [
                            {
                                "name": source_name,
                                "sha256": expected_sha,
                                "source_url": f"{base}/good",
                            }
                        ],
                    }
                ],
            }
            _atomic_json(distribution_path, distribution)
            package_path = root / "fixture-worker.safetensors"
            package_receipt = acquire_worker_package(
                distribution_path,
                "fixture-worker",
                cache,
                package_path,
                retries=2,
                timeout_seconds=5.0,
            )
            fixture_config = root / "config.json"
            fixture_config.write_bytes(b'{"model_type":"fixture"}\n')
            snapshot_path = root / "activated-fixture-worker"
            snapshot = activate_worker_snapshot(
                placement_path,
                "fixture-worker",
                package_path,
                fixture_config,
                snapshot_path,
            )
            snapshot_identity = _read(snapshot_path / "model-identity.json")
            gates = {
                "partial_range_resume_used": bool(first["range_resume_used"]),
                "transient_failure_retried": int(first["attempts"]) == 2,
                "download_hash_exact": str(first["sha256"]) == expected_sha,
                "corruption_rejected": corrupt_rejected,
                "corrupt_object_not_activated": not _object_path(
                    corrupt_cache, expected_sha
                ).exists(),
                "warm_cache_zero_download": bool(second["cache_hit"])
                and int(second["downloaded_bytes"]) == 0,
                "warm_cache_zero_requests": request_count_after_second
                == request_count_after_first,
                "atomic_package_activation": bool(package_receipt["atomic_activation"]),
                "partial_package_absent": not package_path.with_suffix(
                    package_path.suffix + ".partial"
                ).exists(),
                "targeted_worker_scope": bool(package_receipt["targeted_acquisition"]),
                "package_hash_measured": len(
                    str(package_receipt["package"]["package_sha256"])
                )
                == 64,
                "loadable_snapshot_index_exact": int(snapshot["tensor_count"]) == 1,
                "loadable_snapshot_config_exact": str(snapshot["config_sha256"])
                == placement["checkpoint"]["config_sha256"],
                "loadable_snapshot_atomic": bool(snapshot["atomic_activation"]),
                "snapshot_partial_directory_absent": bool(
                    snapshot["partial_directory_absent"]
                ),
                "snapshot_model_identity_exact": (
                    snapshot_identity["schema_version"]
                    == "swarm-model-identity-v1"
                    and snapshot_identity["model_revision"]
                    == "fixture-immutable-revision"
                    and snapshot_identity["adapter_id"] == "kimi_k3_cuda"
                    and snapshot_identity["assignment_sha256"] == "a" * 64
                ),
            }
            receipt.update(
                {
                    "status": "PASS" if all(gates.values()) else "FAIL",
                    "fixture": {
                        "source_bytes": len(payload),
                        "preseeded_partial_bytes": 65536,
                        "expected_sha256": expected_sha,
                        "server_request_counts": dict(_FixtureHandler.counts),
                        "range_headers": list(_FixtureHandler.ranges),
                    },
                    "first_acquisition": first,
                    "warm_cache_acquisition": second,
                    "corruption": {
                        "rejected": corrupt_rejected,
                        "message": corrupt_message,
                    },
                    "package": package_receipt,
                    "snapshot": snapshot,
                    "acceptance_gates": gates,
                    "inspection": {
                        "activated_package_bytes": package_path.stat().st_size,
                        "activated_package_sha256": _sha256(package_path),
                        "source_view_hash_exact": _sha256(
                            cache / "views" / "fixture-worker" / source_name
                        )
                        == expected_sha,
                    },
                    "decision": (
                        "RETAIN_REMOTE_ACQUISITION_CONTRACT"
                        if all(gates.values())
                        else "REDESIGN_REMOTE_ACQUISITION"
                    ),
                }
            )
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
    _atomic_json(output_path, receipt)
    return receipt


def benchmark_kimi_tokenizer_product_seam(
    tokenizer_directory: Path,
    conversation_receipt_path: Path,
    output_path: Path,
    *,
    cycle_id: str = "H014-037b2a",
) -> dict[str, Any]:
    """Activate and execute the exact-hash stage-zero tokenizer boundary."""

    source = tokenizer_directory.resolve()
    conversation_source = conversation_receipt_path.resolve()
    conversation = _read(conversation_source)
    receipt: dict[str, Any] = {
        "schema_version": "experiment-014-k3-tokenizer-product-seam-v1",
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": (
            "An exact-hashed stage-zero tokenizer allowlist makes Kimi product text "
            "tokenization equivalent to the checkpoint tokenizer while every identity "
            "or asset mismatch rejects before custom code executes."
        ),
    }
    try:
        if conversation.get("status") != "PASS":
            raise AcquisitionError("conversation semantics receipt is not passing")
        source_hashes = {
            name: _sha256(source / name) for name in KIMI_TOKENIZER_ASSETS
        }
        expected_conversation_hashes = conversation["checkpoint_sources"]
        lineage_exact = (
            source_hashes["encoding_k3.py"]
            == expected_conversation_hashes["encoding_k3_sha256"]
            and source_hashes["tiktoken.model"]
            == expected_conversation_hashes["tiktoken_model_sha256"]
            and source_hashes["tokenization_kimi.py"]
            == expected_conversation_hashes["tokenization_kimi_sha256"]
        )
        with TemporaryDirectory(prefix="h014-037b2-") as temporary_name:
            root = Path(temporary_name)
            package = root / "stage-zero.safetensors"
            header = {
                "__metadata__": {
                    "worker_id": "k3-worker-000",
                    "checkpoint_fingerprint": "1" * 64,
                },
                "fixture.tensor": {
                    "dtype": "U8",
                    "shape": [1],
                    "data_offsets": [0, 1],
                },
            }
            encoded_header = json.dumps(
                header, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            encoded_header += b" " * ((8 - len(encoded_header) % 8) % 8)
            package.write_bytes(
                _HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + b"\x00"
            )
            placement_path = root / "placement.json"
            placement = {
                "status": "PASS",
                "checkpoint": {
                    "revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
                    "checkpoint_fingerprint": "1" * 64,
                    "config_sha256": _sha256(source / "config.json"),
                },
                "workers": [
                    {
                        "worker_id": "k3-worker-000",
                        "worker_index": 0,
                        "assignment_sha256": "2" * 64,
                        "worker_role": "embedding_dense_stage",
                        "owned_components": ["embedding", "layer_0"],
                        "owned_layers": [0],
                        "source_weight_bytes": 1,
                        "assignment_units": [
                            {
                                "unit_id": "fixture-stage-zero",
                                "tensors": [
                                    {
                                        "name": "fixture.tensor",
                                        "safetensors_file": "fixture.safetensors",
                                        "byte_range": [0, 1],
                                        "physical_bytes": 1,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            _atomic_json(placement_path, placement)
            snapshot = root / "snapshot"
            activation = activate_worker_snapshot(
                placement_path,
                "k3-worker-000",
                package,
                source / "config.json",
                snapshot,
            )
            identity_path = snapshot / "model-identity.json"
            verify_pinned_kimi_tokenizer_assets(
                snapshot,
                identity_path,
                expected_worker_id="k3-worker-000",
            )
            snapshot_tokenizer = load_pinned_kimi_tokenizer(
                snapshot,
                identity_path,
                expected_worker_id="k3-worker-000",
            )
            from transformers import AutoTokenizer

            source_tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                source,
                local_files_only=True,
                trust_remote_code=True,
            )
            prompts = [
                "Hi",
                "Hypothesis → Benchmark",
                "Can a worker store less than one complete Kimi layer?",
            ]
            cases: list[dict[str, Any]] = []
            for prompt in prompts:
                expected_raw = [
                    int(value)
                    for value in source_tokenizer(
                        prompt, add_special_tokens=False, return_tensors=None
                    )["input_ids"]
                ]
                observed_raw = [
                    int(value)
                    for value in snapshot_tokenizer(
                        prompt, add_special_tokens=False, return_tensors=None
                    )["input_ids"]
                ]
                expected = apply_kimi_prompt_special_tokens(
                    source_tokenizer,
                    expected_raw,
                    add_special_tokens=True,
                )
                observed = apply_kimi_prompt_special_tokens(
                    snapshot_tokenizer,
                    observed_raw,
                    add_special_tokens=True,
                )
                cases.append(
                    {
                        "prompt": prompt,
                        "raw_token_ids": observed_raw,
                        "token_count": len(observed),
                        "token_ids": observed,
                        "token_sha256": hashlib.sha256(
                            json.dumps(observed, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "raw_exact": observed_raw == expected_raw,
                        "exact": observed == expected,
                    }
                )
            identity = _read(identity_path)
            controls: dict[str, dict[str, Any]] = {}

            def rejected(name: str, callable_: Any) -> bool:
                try:
                    callable_()
                except (IntegrityError, AcquisitionError, ValueError) as exc:
                    controls[name] = {
                        "rejected": True,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                    return True
                controls[name] = {
                    "rejected": False,
                    "detail": "candidate was incorrectly accepted",
                }
                return False

            wrong_worker = rejected(
                "wrong_worker",
                lambda: verify_pinned_kimi_tokenizer_assets(
                    snapshot,
                    identity_path,
                    expected_worker_id="k3-worker-001",
                ),
            )
            asset = snapshot / "encoding_k3.py"
            original_asset = asset.read_bytes()
            asset.write_bytes(original_asset + b"\n# tampered\n")
            tampered_asset = rejected(
                "tampered_asset",
                lambda: verify_pinned_kimi_tokenizer_assets(
                    snapshot,
                    identity_path,
                    expected_worker_id="k3-worker-000",
                ),
            )
            asset.write_bytes(original_asset)
            missing_asset_path = snapshot / "tiktoken.model"
            missing_asset = missing_asset_path.read_bytes()
            missing_asset_path.unlink()
            absent_asset = rejected(
                "missing_asset",
                lambda: verify_pinned_kimi_tokenizer_assets(
                    snapshot,
                    identity_path,
                    expected_worker_id="k3-worker-000",
                ),
            )
            missing_asset_path.write_bytes(missing_asset)
            non_owner_identity = {**identity, "owns_embeddings": False}
            _atomic_json(identity_path, non_owner_identity)
            non_owner = rejected(
                "non_stage_zero_owner",
                lambda: verify_pinned_kimi_tokenizer_assets(
                    snapshot,
                    identity_path,
                    expected_worker_id="k3-worker-000",
                ),
            )
            expanded_identity = dict(identity)
            expanded_identity["tokenizer_assets_sha256"] = {
                **identity["tokenizer_assets_sha256"],
                "unexpected.py": "0" * 64,
            }
            _atomic_json(identity_path, expanded_identity)
            expanded_allowlist = rejected(
                "expanded_allowlist",
                lambda: verify_pinned_kimi_tokenizer_assets(
                    snapshot,
                    identity_path,
                    expected_worker_id="k3-worker-000",
                ),
            )
            _atomic_json(identity_path, identity)
            hi_raw = list(cases[0]["raw_token_ids"])
            hi_with_bos = list(cases[0]["token_ids"])
            invalid_bos = rejected(
                "invalid_bos",
                lambda: apply_kimi_prompt_special_tokens(
                    object(), hi_raw, add_special_tokens=True
                ),
            )
            gates = {
                "conversation_asset_lineage_exact": lineage_exact,
                "stage_zero_activation_copied_exact_allowlist": (
                    int(activation["tokenizer_asset_count"])
                    == len(KIMI_TOKENIZER_ASSETS)
                    and activation["tokenizer_assets_sha256"] == source_hashes
                ),
                "all_raw_prompt_token_ids_exact": all(
                    case["raw_exact"] for case in cases
                ),
                "all_product_prompt_token_ids_exact": all(
                    case["exact"] for case in cases
                ),
                "oracle_hi_prompt_token_count_two": cases[0]["token_count"] == 2,
                "oracle_hi_exact_bos_contract": hi_with_bos == [163584, 18699],
                "bos_inserted_exactly_once": apply_kimi_prompt_special_tokens(
                    snapshot_tokenizer,
                    hi_with_bos,
                    add_special_tokens=True,
                )
                == hi_with_bos,
                "special_tokens_false_preserves_raw_ids": (
                    apply_kimi_prompt_special_tokens(
                        snapshot_tokenizer,
                        hi_raw,
                        add_special_tokens=False,
                    )
                    == hi_raw
                ),
                "invalid_bos_rejected": invalid_bos,
                "wrong_worker_rejected": wrong_worker,
                "tampered_asset_rejected": tampered_asset,
                "missing_asset_rejected": absent_asset,
                "non_stage_zero_owner_rejected": non_owner,
                "expanded_allowlist_rejected": expanded_allowlist,
                "post_control_identity_restored": bool(
                    verify_pinned_kimi_tokenizer_assets(
                        snapshot,
                        identity_path,
                        expected_worker_id="k3-worker-000",
                    )
                ),
            }
            receipt.update(
                {
                    "status": "PASS" if all(gates.values()) else "FAIL",
                    "sources": {
                        "tokenizer_directory": str(source),
                        "conversation_receipt": str(conversation_source),
                        "conversation_receipt_sha256": _sha256(conversation_source),
                        "tokenizer_assets_sha256": source_hashes,
                    },
                    "activation": activation,
                    "token_cases": cases,
                    "negative_controls": controls,
                    "acceptance_gates": gates,
                    "inspection": (
                        "Only four hash-pinned local tokenizer assets were trusted. "
                        "The repository-owned Kimi adapter then reproduced the native "
                        "engine's explicit, exactly-once BOS prompt contract."
                    ),
                    "bottleneck": (
                        "Tokenizer construction is a stage-zero cold lifecycle cost; it is "
                        "cached after first use and does not affect warm per-token CUDA service."
                    ),
                    "decision": "RETAIN_HASH_PINNED_TOKENIZER_AND_EXPLICIT_BOS_ADAPTER",
                    "redesign": (
                        "Package the same four immutable assets with worker zero only and "
                        "exercise text submission on the physical canary."
                    ),
                }
            )
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    _atomic_json(output_path, receipt)
    return receipt


__all__ = [
    "AcquisitionError",
    "acquire_worker_package",
    "activate_worker_snapshot",
    "benchmark_kimi_tokenizer_product_seam",
    "benchmark_remote_acquisition_fixture",
]
