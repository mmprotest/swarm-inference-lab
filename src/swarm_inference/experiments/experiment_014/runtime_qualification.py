"""Cross-platform native-source and physical Linux runtime identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "experiment-015-k3-native-source-manifest-v1"
CERTIFICATE_SCHEMA = "experiment-015-k3-linux-runtime-qualification-v1"
NATIVE_SOURCE_FILES = (
    "backend_cuda.cu",
    "backend_cuda.h",
    "backend_gpu_compat.h",
)
KIMI_OPERATION_CLASSES = (
    "mxfp4_routed_expert",
    "router_top16",
    "grouped_int4_dense",
    "final_rmsnorm",
    "embedding",
    "lm_head",
    "shared_expert",
    "moe_reduction",
    "attnres",
    "kda_stage",
    "gated_mla_stage",
)
KIMI_STAGE_ROLES = (
    "stage_zero_embedding_dense",
    "kda_moe",
    "gated_mla_moe",
    "final_norm_head_sampling",
)
PHYSICAL_CERTIFICATE_GATES = (
    "exact_rtx_3090",
    "compute_capability_sm86",
    "vram_at_least_24_gib",
    "linux_elf_x86_64",
    "binary_hash_exact",
    "sm86_sass_present",
    "compute86_ptx_present",
    "all_11_kimi_operation_classes_pass",
    "all_four_stage_roles_pass",
    "full_graph_correctness_reference_pass",
    "stateful_decode_pass",
    "production_batch_8_pass",
    "over_limit_batch_rejected_before_cuda",
    "prepare_exactly_seven_calls",
    "warm_lifecycle_deltas_zero",
    "cuda_error_state_clear",
    "post_canary_safe_fixture_pass",
    "nvidia_smi_healthy_after",
)
BUILD_ARGUMENTS = (
    "nvcc",
    "-O3",
    "-std=c++17",
    "-shared",
    "-Xcompiler=-fPIC,-Wall,-Wextra",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_86,code=compute_86",
    "-DCOLI_CUDA_BUILDING_DLL",
    "-DCOLI_CUDA_MIN_CC=86",
    "-DCOLI_CUDA_HAS_FORWARD_PTX=1",
    "backend_cuda.cu",
    "-lcudart",
    "-o",
    "libcoli_cuda-sm86.so",
)


class RuntimeQualificationError(RuntimeError):
    """A Linux runtime lacks exact source or physical-canary provenance."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeQualificationError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeQualificationError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeQualificationError(f"{name} must be 64 lowercase hexadecimal digits")
    return text


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def build_native_source_manifest(
    source_directory: Path,
    placement_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind the exact portable CUDA sources and compiler contract."""

    source = source_directory.resolve()
    placement_source = placement_path.resolve()
    placement = _read(placement_source)
    if placement.get("status") != "PASS":
        raise RuntimeQualificationError("placement is not passing")
    files: dict[str, dict[str, Any]] = {}
    for name in NATIVE_SOURCE_FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeQualificationError(f"native source is absent or linked: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    bundle_identity = [
        {"name": name, **files[name]} for name in sorted(NATIVE_SOURCE_FILES)
    ]
    manifest = {
        "schema_version": SOURCE_SCHEMA,
        "status": "PASS",
        "target": "linux-x86_64 CUDA sm_86 SASS plus compute_86 PTX",
        "source_files": files,
        "source_bundle_sha256": _canonical_sha256(bundle_identity),
        "build_arguments": list(BUILD_ARGUMENTS),
        "placement_sha256": _sha256(placement_source),
        "windows_reference_sha256": _require_sha256(
            placement["runtime"]["cuda_library_sha256"],
            name="Windows reference SHA-256",
        ),
        "binary_output_name": "libcoli_cuda-sm86.so",
        "physical_qualification_required": True,
    }
    _atomic_json(output_path, manifest)
    return {**manifest, "output_sha256": _sha256(output_path.resolve())}


def validate_native_source_manifest(
    source_manifest_path: Path,
    placement_path: Path,
    *,
    source_directory: Path | None = None,
) -> dict[str, Any]:
    manifest_path = source_manifest_path.resolve()
    placement_source = placement_path.resolve()
    manifest = _read(manifest_path)
    placement = _read(placement_source)
    if manifest.get("schema_version") != SOURCE_SCHEMA or manifest.get("status") != "PASS":
        raise RuntimeQualificationError("native source manifest is not passing")
    if manifest.get("build_arguments") != list(BUILD_ARGUMENTS):
        raise RuntimeQualificationError("native build arguments differ from the contract")
    if manifest.get("placement_sha256") != _sha256(placement_source):
        raise RuntimeQualificationError("native source manifest targets another placement")
    if manifest.get("windows_reference_sha256") != placement["runtime"][
        "cuda_library_sha256"
    ]:
        raise RuntimeQualificationError("native source manifest has another Windows reference")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or sorted(source_files) != sorted(
        NATIVE_SOURCE_FILES
    ):
        raise RuntimeQualificationError("native source file allowlist is incomplete or expanded")
    identity: list[dict[str, Any]] = []
    for name in sorted(NATIVE_SOURCE_FILES):
        row = source_files[name]
        if not isinstance(row, dict):
            raise RuntimeQualificationError(f"native source row is invalid: {name}")
        digest = _require_sha256(row.get("sha256"), name=f"native source {name}")
        size = int(row.get("bytes", -1))
        if size <= 0:
            raise RuntimeQualificationError(f"native source size is invalid: {name}")
        identity.append({"name": name, "bytes": size, "sha256": digest})
        if source_directory is not None:
            actual = source_directory.resolve() / name
            if (
                not actual.is_file()
                or actual.is_symlink()
                or actual.stat().st_size != size
                or _sha256(actual) != digest
            ):
                raise RuntimeQualificationError(f"packaged native source differs: {name}")
    if manifest.get("source_bundle_sha256") != _canonical_sha256(identity):
        raise RuntimeQualificationError("native source bundle fingerprint differs")
    return manifest


def validate_linux_runtime_certificate(
    certificate_path: Path,
    source_manifest_path: Path,
    placement_path: Path,
    *,
    allow_logical_fixture: bool = False,
) -> dict[str, str]:
    """Return worker-local ELF identity only after full provenance checks."""

    certificate_source = certificate_path.resolve()
    source_manifest_source = source_manifest_path.resolve()
    placement_source = placement_path.resolve()
    certificate = _read(certificate_source)
    source_manifest = validate_native_source_manifest(
        source_manifest_source, placement_source
    )
    if (
        certificate.get("schema_version") != CERTIFICATE_SCHEMA
        or certificate.get("status") != "PASS"
    ):
        raise RuntimeQualificationError("Linux runtime certificate is not passing")
    evidence_kind = certificate.get("evidence_kind")
    expected_kind = "logical_fixture" if allow_logical_fixture else "physical_3090"
    if evidence_kind != expected_kind:
        raise RuntimeQualificationError(
            f"Linux runtime certificate is not {expected_kind} evidence"
        )
    if certificate.get("platform") != "linux-x86_64":
        raise RuntimeQualificationError("qualified runtime platform is not linux-x86_64")
    references = certificate.get("references")
    if not isinstance(references, dict):
        raise RuntimeQualificationError("qualified runtime references are absent")
    expected_references = {
        "placement_sha256": _sha256(placement_source),
        "source_manifest_sha256": _sha256(source_manifest_source),
        "source_bundle_sha256": source_manifest["source_bundle_sha256"],
        "windows_reference_sha256": source_manifest["windows_reference_sha256"],
    }
    for name, expected in expected_references.items():
        if references.get(name) != expected:
            raise RuntimeQualificationError(f"qualified runtime {name} differs")
    if certificate.get("build_arguments") != source_manifest["build_arguments"]:
        raise RuntimeQualificationError("qualified runtime build arguments differ")
    binary = certificate.get("binary")
    if not isinstance(binary, dict):
        raise RuntimeQualificationError("qualified runtime binary identity is absent")
    binary_path = str(binary.get("worker_local_path", "")).strip()
    binary_sha = _require_sha256(binary.get("sha256"), name="qualified ELF SHA-256")
    if not binary_path.startswith("/") or not binary_path.endswith(".so"):
        raise RuntimeQualificationError("qualified ELF worker-local path is invalid")
    if binary_sha == source_manifest["windows_reference_sha256"]:
        raise RuntimeQualificationError("Linux ELF incorrectly reuses the Windows DLL SHA")
    if int(binary.get("bytes", 0)) <= 0 or binary.get("elf_magic") is not True:
        raise RuntimeQualificationError("qualified binary is not a measured ELF")
    if sorted(certificate.get("operation_classes", [])) != sorted(
        KIMI_OPERATION_CLASSES
    ):
        raise RuntimeQualificationError("qualified runtime lacks an exact operation matrix")
    if sorted(certificate.get("stage_roles", [])) != sorted(KIMI_STAGE_ROLES):
        raise RuntimeQualificationError("qualified runtime lacks an exact stage-role matrix")
    gates = certificate.get("acceptance_gates")
    if not isinstance(gates, dict) or sorted(gates) != sorted(PHYSICAL_CERTIFICATE_GATES):
        raise RuntimeQualificationError("qualified runtime gate set is incomplete or expanded")
    failed = sorted(name for name, value in gates.items() if value is not True)
    if failed:
        raise RuntimeQualificationError(f"qualified runtime has failed gates: {failed}")
    return {
        "worker_local_path": binary_path,
        "sha256": binary_sha,
        "certificate_sha256": _sha256(certificate_source),
        "source_manifest_sha256": _sha256(source_manifest_source),
        "evidence_kind": str(evidence_kind),
    }


__all__ = [
    "BUILD_ARGUMENTS",
    "CERTIFICATE_SCHEMA",
    "KIMI_OPERATION_CLASSES",
    "KIMI_STAGE_ROLES",
    "NATIVE_SOURCE_FILES",
    "PHYSICAL_CERTIFICATE_GATES",
    "RuntimeQualificationError",
    "build_native_source_manifest",
    "validate_linux_runtime_certificate",
    "validate_native_source_manifest",
]
