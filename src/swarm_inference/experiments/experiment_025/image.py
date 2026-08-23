"""Build, inspect, push, and freeze the immutable E025 Linux image."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_json, sha256_file, utc_now

SCHEMA_VERSION = "experiment-025-deployment-image-v2"

_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_BEARER_PARAMETER = re.compile(r'([A-Za-z][A-Za-z0-9_-]*)="([^"\\]*)"')
_MAX_TOKEN_RESPONSE_BYTES = 1 << 20
_MAX_MANIFEST_BYTES = 16 << 20


def _run(
    arguments: list[str],
    *,
    repo: Path,
    timeout_seconds: float,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=timeout_seconds,
        check=False,
    )


def _inspect(repo: Path, reference: str) -> dict[str, Any]:
    result = _run(
        ["docker", "image", "inspect", reference],
        repo=repo,
        timeout_seconds=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker image inspect failed: {result.stderr[-1000:]}")
    value = json.loads(result.stdout)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RuntimeError("docker image inspect returned an unexpected payload")
    row = value[0]
    return {
        "id": row.get("Id"),
        "repo_tags": row.get("RepoTags", []),
        "repo_digests": row.get("RepoDigests", []),
        "created": row.get("Created"),
        "architecture": row.get("Architecture"),
        "os": row.get("Os"),
        "size_bytes": row.get("Size"),
        "labels": row.get("Config", {}).get("Labels", {}),
    }


def _read_http_response(
    opener: Any,
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    maximum_bytes: int,
) -> tuple[int, Any, bytes]:
    with opener(request, timeout=timeout_seconds) as response:
        body = response.read(maximum_bytes + 1)
        if len(body) > maximum_bytes:
            raise ValueError("anonymous registry response exceeded its byte limit")
        status = int(getattr(response, "status", response.getcode()))
        return status, response.headers, body


def _bearer_parameters(challenge: str | None) -> dict[str, str]:
    if challenge is None or not challenge.lower().startswith("bearer "):
        return {}
    return {
        match.group(1).lower(): match.group(2)
        for match in _BEARER_PARAMETER.finditer(challenge[7:])
    }


def anonymous_registry_manifest_probe(
    reference: str,
    *,
    timeout_seconds: float = 60,
    opener: Any | None = None,
) -> dict[str, Any]:
    """Resolve an immutable OCI digest without ambient Docker credentials.

    A public registry may issue a short-lived bearer token to an unauthenticated
    caller.  This probe follows only that standard challenge flow and never
    consults Docker's credential store, the local engine, or environment auth.
    """

    if "@" not in reference:
        raise ValueError("anonymous registry probe requires an immutable reference")
    registry_repository, digest = reference.rsplit("@", maxsplit=1)
    if "/" not in registry_repository:
        raise ValueError("registry reference must contain a repository path")
    registry, repository = registry_repository.split("/", maxsplit=1)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("registry reference has an invalid sha256 digest")
    if not registry or not repository:
        raise ValueError("registry reference is incomplete")

    open_url = opener or urllib.request.urlopen
    manifest_url = f"https://{registry}/v2/{repository}/manifests/{digest}"
    base: dict[str, Any] = {
        "schema_version": "experiment-025-anonymous-registry-probe-v1",
        "probe_method": "RAW_OCI_ANONYMOUS_BEARER_CHALLENGE",
        "registry": registry,
        "repository": repository,
        "requested_digest": digest,
        "credentials_supplied": False,
        "docker_credential_store_consulted": False,
        "local_docker_engine_consulted": False,
        "token_persisted": False,
    }

    def failed(stage: str, error: Exception, status_code: int | None) -> dict[str, Any]:
        return {
            **base,
            "status": "FAIL",
            "failed_stage": stage,
            "http_status": status_code,
            "error": (
                f"HTTP {status_code}: {getattr(error, 'reason', '')}".strip()
                if status_code is not None
                else f"{type(error).__name__}: {error}"
            ),
        }

    def verified(
        *,
        status_code: int,
        headers: Any,
        body: bytes,
        authentication: str,
        initial_status: int,
        token_status: int | None,
    ) -> dict[str, Any]:
        header_digest = str(headers.get("Docker-Content-Digest", "")).lower()
        computed_digest = "sha256:" + hashlib.sha256(body).hexdigest()
        digest_match = digest in {header_digest, computed_digest}
        return {
            **base,
            "status": "PASS" if status_code == 200 and digest_match else "FAIL",
            "authentication": authentication,
            "initial_http_status": initial_status,
            "anonymous_token_http_status": token_status,
            "manifest_http_status": status_code,
            "manifest_content_type": str(headers.get("Content-Type", "")),
            "manifest_bytes": len(body),
            "header_digest": header_digest or None,
            "computed_digest": computed_digest,
            "digest_match": digest_match,
        }

    manifest_request = urllib.request.Request(
        manifest_url,
        headers={"Accept": _MANIFEST_ACCEPT, "User-Agent": "swarm-e025-anonymous-probe"},
        method="GET",
    )
    try:
        status, headers, body = _read_http_response(
            open_url,
            manifest_request,
            timeout_seconds=timeout_seconds,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    except urllib.error.HTTPError as error:
        if error.code != 401:
            return failed("initial_manifest_request", error, error.code)
        initial_status = error.code
        parameters = _bearer_parameters(error.headers.get("WWW-Authenticate"))
    except Exception as error:  # pragma: no cover - exact network exception is platform-specific
        return failed("initial_manifest_request", error, None)
    else:
        return verified(
            status_code=status,
            headers=headers,
            body=body,
            authentication="none",
            initial_status=status,
            token_status=None,
        )

    realm = parameters.get("realm", "")
    realm_parts = urllib.parse.urlsplit(realm)
    if realm_parts.scheme != "https" or not realm_parts.netloc:
        return failed(
            "bearer_challenge",
            ValueError("registry did not provide an HTTPS bearer realm"),
            initial_status,
        )
    token_query = urllib.parse.parse_qsl(realm_parts.query, keep_blank_values=True)
    service = parameters.get("service")
    if service:
        token_query.append(("service", service))
    token_query.append(("scope", f"repository:{repository}:pull"))
    token_url = urllib.parse.urlunsplit(
        (
            realm_parts.scheme,
            realm_parts.netloc,
            realm_parts.path,
            urllib.parse.urlencode(token_query),
            "",
        )
    )
    token_request = urllib.request.Request(
        token_url,
        headers={"Accept": "application/json", "User-Agent": "swarm-e025-anonymous-probe"},
        method="GET",
    )
    try:
        token_status, _token_headers, token_body = _read_http_response(
            open_url,
            token_request,
            timeout_seconds=timeout_seconds,
            maximum_bytes=_MAX_TOKEN_RESPONSE_BYTES,
        )
        token_payload = json.loads(token_body)
        token = token_payload.get("token") or token_payload.get("access_token")
        if token_status != 200 or not isinstance(token, str) or not token:
            raise ValueError("anonymous registry token response did not contain a token")
    except urllib.error.HTTPError as error:
        return failed("anonymous_token_request", error, error.code)
    except Exception as error:
        return failed("anonymous_token_request", error, None)

    authenticated_manifest_request = urllib.request.Request(
        manifest_url,
        headers={
            "Accept": _MANIFEST_ACCEPT,
            "Authorization": f"Bearer {token}",
            "User-Agent": "swarm-e025-anonymous-probe",
        },
        method="GET",
    )
    try:
        status, headers, body = _read_http_response(
            open_url,
            authenticated_manifest_request,
            timeout_seconds=timeout_seconds,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    except urllib.error.HTTPError as error:
        return failed("anonymous_manifest_request", error, error.code)
    except Exception as error:  # pragma: no cover - exact network exception is platform-specific
        return failed("anonymous_manifest_request", error, None)
    return verified(
        status_code=status,
        headers=headers,
        body=body,
        authentication="anonymous_bearer_token",
        initial_status=initial_status,
        token_status=token_status,
    )


def build_and_publish_image(
    *,
    repo: Path,
    image_repository: str,
    tag: str,
    source_id: str,
    output_path: Path,
    push: bool = True,
) -> dict[str, Any]:
    root = repo.expanduser().resolve()
    dockerfile = root / "deployment" / "Dockerfile.e025"
    image_context = root / "deployment" / "e025_context"
    if not dockerfile.is_file() or not (image_context / "index.json").is_file():
        raise ValueError("E025 Dockerfile or frozen image inputs are absent")
    reference = f"{image_repository}:{tag}"
    build = _run(
        [
            "docker",
            "build",
            "--pull",
            "--platform",
            "linux/amd64",
            "--file",
            str(dockerfile),
            "--build-arg",
            f"E025_SOURCE_ID={source_id}",
            "--tag",
            reference,
            ".",
        ],
        repo=root,
        timeout_seconds=5400,
    )
    build_pass = build.returncode == 0
    local = _inspect(root, reference) if build_pass else None
    push_result: subprocess.CompletedProcess[str] | None = None
    if build_pass and push:
        push_result = _run(
            ["docker", "push", reference],
            repo=root,
            timeout_seconds=3600,
        )
    push_pass = bool(push_result is not None and push_result.returncode == 0)
    remote = _inspect(root, reference) if push_pass else local
    repo_digests = list(remote.get("repo_digests", [])) if remote else []
    matching = [
        str(value).split("@", maxsplit=1)[1]
        for value in repo_digests
        if str(value).startswith(image_repository + "@sha256:")
    ]
    immutable_digest = matching[0] if len(set(matching)) == 1 else None
    anonymous_probe: dict[str, Any] | None = None
    if push_pass and immutable_digest is not None:
        anonymous_probe = anonymous_registry_manifest_probe(
            f"{image_repository}@{immutable_digest}", timeout_seconds=180
        )
    anonymous_pass = bool(
        anonymous_probe is not None and anonymous_probe.get("status") == "PASS"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": (
            "PASS"
            if build_pass
            and (push_pass if push else True)
            and immutable_digest is not None
            and (anonymous_pass if push else True)
            else "FAIL"
        ),
        "reference": reference,
        "immutable_reference": (
            f"{image_repository}@{immutable_digest}" if immutable_digest else None
        ),
        "immutable_digest": immutable_digest,
        "source_id": source_id,
        "dockerfile": str(dockerfile),
        "dockerfile_sha256": sha256_file(dockerfile),
        "image_input_index_sha256": sha256_file(image_context / "index.json"),
        "base_images": {
            "devel": "nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92",
            "runtime": "nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:c3fde347d52d578c84fd644bc177bc7ec333feaf11550d990da4084d7612e4c7",
        },
        "platform": "linux/amd64",
        "native_architecture": "sm_86+sm_89+sm_120+compute_120_ptx",
        "native_consumer_targets": ["sm_86", "sm_89", "sm_120"],
        "build_status": "PASS" if build_pass else "FAIL",
        "push_requested": push,
        "push_status": "PASS" if push_pass else "FAIL",
        "anonymous_registry_resolve_status": (
            "PASS" if anonymous_pass else "FAIL"
        ),
        "anonymous_registry_probe": anonymous_probe,
        "local_inspect": local,
        "remote_inspect": remote,
        "build_stdout_tail": build.stdout[-4000:],
        "build_stderr_tail": build.stderr[-4000:],
        "push_stdout_tail": push_result.stdout[-4000:] if push_result else "",
        "push_stderr_tail": push_result.stderr[-4000:] if push_result else "",
        "secrets_logged": False,
    }
    atomic_write_json(output_path, payload)
    return payload


def validate_local_5090_image(
    *,
    repo: Path,
    image_receipt_path: Path,
    checkpoint: Path,
    oracle_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Physically execute the frozen Linux native binary on the local RTX 5090."""

    root = repo.expanduser().resolve()
    image = read_json(image_receipt_path.expanduser().resolve())
    immutable_reference = str(image.get("immutable_reference", ""))
    immutable_digest = str(image.get("immutable_digest", ""))
    if image.get("status") != "PASS" or "@sha256:" not in immutable_reference:
        raise ValueError("E025 local image canary requires a published immutable image")
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        repo=root,
        timeout_seconds=30,
    )
    fields = [value.strip() for value in gpu.stdout.strip().split(",")]
    physical_gpu = {
        "gpu_name": fields[0] if len(fields) == 5 else "unknown",
        "gpu_uuid": fields[1] if len(fields) == 5 else "unknown",
        "vram_mib": int(fields[2]) if len(fields) == 5 else 0,
        "driver_version": fields[3] if len(fields) == 5 else "unknown",
        "compute_capability": fields[4] if len(fields) == 5 else "unknown",
    }
    container_output = output_path.with_suffix(".container.json").resolve()
    checkpoint_root = checkpoint.expanduser().resolve()
    oracle = oracle_root.expanduser().resolve()
    python_code = (
        "from pathlib import Path; "
        "from swarm_inference.experiments.experiment_014.full_cuda import "
        "benchmark_streamed_cuda_graph; "
        "benchmark_streamed_cuda_graph(Path('/checkpoint'),"
        "Path('/opt/swarm/native/libcoli_cuda-consumer.so'),"
        "Path('/oracle/hidden-trace.f32'),Path('/oracle/routes.txt'),"
        f"Path('/evidence/{container_output.name}'),"
        "layer_limit=4,prompt_token_ids=(163584,18699),decode_token_id=11,"
        "device=0,relative_error_gate=3e-3,cycle_id='E025-LOCAL-IMAGE-5090',"
        "oracle_layer_count=93)"
    )
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "device=0",
            "--network",
            "none",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=bind,source={checkpoint_root},target=/checkpoint,readonly",
            "--mount",
            f"type=bind,source={oracle},target=/oracle,readonly",
            "--mount",
            f"type=bind,source={container_output.parent},target=/evidence",
            "--entrypoint",
            "python",
            immutable_reference,
            "-c",
            python_code,
        ],
        repo=root,
        timeout_seconds=900,
    )
    benchmark = read_json(container_output) if container_output.is_file() else None
    gates = {
        "physical_local_gpu_is_consumer_5090": "GEFORCE RTX 5090"
        in physical_gpu["gpu_name"].upper(),
        "frozen_immutable_image": immutable_digest.startswith("sha256:")
        and immutable_reference.endswith(immutable_digest),
        "linux_container_saw_physical_gpu": result.returncode == 0,
        "current_native_binary_executed": benchmark is not None
        and benchmark.get("status") == "PASS",
        "real_kimi_checkpoint_fixture_passed": benchmark is not None
        and benchmark.get("status") == "PASS",
        "headline_evidence_not_claimed": True,
    }
    payload = {
        "schema_version": "experiment-025-local-5090-image-canary-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "evidence_class": "PHYSICAL_SINGLE_MACHINE_LOCAL_CONSUMER_GPU_CANARY",
        "immutable_reference": immutable_reference,
        "immutable_digest": immutable_digest,
        "physical_gpu": physical_gpu,
        "benchmark": benchmark,
        "gates": gates,
        "docker_returncode": result.returncode,
        "docker_stdout_tail": result.stdout[-2000:],
        "docker_stderr_tail": result.stderr[-4000:],
        "vast_mutations_performed": False,
        "permitted_use": "runtime compatibility gate only; not headline swarm evidence",
    }
    atomic_write_json(output_path, payload)
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "anonymous_registry_manifest_probe",
    "build_and_publish_image",
    "validate_local_5090_image",
]
