"""Mutation-locked future RunPod canary and headline operator for E025."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
import socket
import ssl
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_json, utc_now
from .providers.base import EndpointMode, ProviderMutationPolicy, ProviderOperation
from .providers.runpod import (
    RunPodProvider,
    UrllibRunPodTransport,
    allocation_dict,
    normalize_pod,
    redact_provider_payload,
    resolve_endpoint,
)
from .runpod_cleanup import (
    AppendOnlyRunPodLedger,
    cleanup_from_ledger,
    start_runpod_watchdog,
)
from .runpod_planning import (
    BACKBONE_GPU_ID,
    FRAGMENT_GPU_IDS,
    IMAGE_TAG,
    RUN_ID,
    encode_worker_specs,
)
from .secrets import create_transport_material

PAID_MODES = frozenset({"p1", "p2", "p3", "p4", "p5", "headline"})
LIGHTWEIGHT_MODES = frozenset({"p2", "p5"})


def _manifests(preflight: Path) -> list[dict[str, Any]]:
    value = read_json(preflight / "runpod-pod-role-manifests.json")
    manifests = value.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("RunPod Pod role manifests are absent")
    return manifests


def _first_multi_gpu(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        row
        for row in manifests
        if row["pod_class"] == "BACKBONE" and int(row["requested_gpu_count"]) > 1
    )


def select_stage_manifests(manifests: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    fragments = [row for row in manifests if row["pod_class"] == "FRAGMENT"]
    parent = [row for row in manifests if row["pod_class"] == "LAYER89_PARENT"]
    backbone = [row for row in manifests if row["pod_class"] == "BACKBONE"]
    if mode == "p1":
        first = copy.deepcopy(
            next(row for row in backbone if any(int(role["layer"]) == 1 for role in row["roles"]))
        )
        role_index = next(
            index for index, role in enumerate(first["roles"]) if int(role["layer"]) == 1
        )
        first["logical_pod_id"] = f"e025-rp-{RUN_ID}-p1-single"
        first["pod_name"] = first["logical_pod_id"]
        first["roles"] = [first["roles"][role_index]]
        worker_spec = copy.deepcopy(first["worker_specs"][role_index])
        worker_spec["gpu_slot"] = 0
        worker_spec["port"] = 42525
        first["worker_specs"] = [worker_spec]
        first["requested_gpu_count"] = 1
        first["gpu_slots"] = [0]
        first["internal_serving_ports"] = [42525]
        first["E025_WORKER_SPECS_B64"] = encode_worker_specs(first["worker_specs"])
        return [first]
    if mode == "p2":
        selected: list[dict[str, Any]] = []
        for index in range(2):
            item = copy.deepcopy(select_stage_manifests(manifests, "p1")[0])
            item["logical_pod_id"] = f"e025-rp-{RUN_ID}-p2-network-{index}"
            item["pod_name"] = item["logical_pod_id"]
            item["pod_class"] = "NETWORK_CANARY"
            selected.append(item)
        return selected
    if mode == "p3":
        item = copy.deepcopy(_first_multi_gpu(manifests))
        item["logical_pod_id"] = f"e025-rp-{RUN_ID}-p3-multigpu"
        item["pod_name"] = item["logical_pod_id"]
        return [item]
    if mode == "p4":
        return [*fragments, *parent]
    if mode in {"p5", "headline"}:
        return [*fragments, *parent, *backbone]
    raise ValueError(f"unsupported RunPod paid stage: {mode}")


def choose_fragment_gpu(preflight: Path) -> str | None:
    inventory = read_json(preflight / "runpod-live-gpu-inventory.json")
    rows = {str(row["gpu_type_id"]): row for row in inventory.get("gpu_types", [])}
    for gpu_id in FRAGMENT_GPU_IDS:
        row = rows.get(gpu_id, {})
        located = [
            dc
            for dc in row.get("datacenter_availability", [])
            if str(dc.get("stockStatus", "")).lower() not in {"", "none"}
        ]
        counts = (
            row.get("graphql", {})
            .get("tiers", {})
            .get("COMMUNITY", {})
            .get("schedulable_gpu_counts_inferred_from_non_null_lowest_price", [])
        )
        if located and 1 in counts:
            return gpu_id
    return None


def _secret_values(material: dict[str, Any]) -> dict[str, str]:
    def encoded(name: str) -> str:
        return base64.b64encode(Path(str(material[name])).read_bytes()).decode("ascii")

    return {
        "E025_RUN_CREDENTIAL_B64": encoded("credential_path"),
        "E025_TLS_CERT_B64": encoded("certificate_path"),
        "E025_TLS_KEY_B64": encoded("private_key_path"),
    }


def _inline_python(source: str) -> str:
    encoded = base64.b64encode(textwrap.dedent(source).encode("utf-8")).decode("ascii")
    return (
        f"import base64;exec(compile(base64.b64decode('{encoded}'),'<e025-runpod-inline>','exec'))"
    )


def lightweight_start_code(*, network_probe: bool = False) -> str:
    """No-K3 allocation barrier or authenticated P2 TLS probe service."""

    if not network_probe:
        return _inline_python(
            """
            import json, os, socket, threading, time

            ports = [int(value) for value in os.environ.get("E025_LIGHTWEIGHT_PORTS", "42525").split(",")]

            def serve(listener):
                while True:
                    connection, _ = listener.accept()
                    connection.close()

            for port in ports:
                listener = socket.socket()
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("0.0.0.0", port))
                listener.listen(32)
                threading.Thread(target=serve, args=(listener,), daemon=True).start()
            print(json.dumps({"event": "E025_LIGHTWEIGHT_READY", "ports": ports}), flush=True)
            time.sleep(21600)
            """
        )
    return _inline_python(
        """
        import base64, hashlib, hmac, json, os, socket, ssl, tempfile, threading, time

        credential = base64.b64decode(os.environ["E025_RUN_CREDENTIAL_B64"], validate=True)
        certificate = base64.b64decode(os.environ["E025_TLS_CERT_B64"], validate=True)
        private_key = base64.b64decode(os.environ["E025_TLS_KEY_B64"], validate=True)
        root = tempfile.mkdtemp(prefix="e025-p2-")
        cert_path, key_path = root + "/server.crt", root + "/server.key"
        open(cert_path, "wb").write(certificate)
        open(key_path, "wb").write(private_key)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.minimum_version = ssl.TLSVersion.TLSv1_2
        server_context.load_cert_chain(cert_path, key_path)

        def exact(channel, size):
            chunks, remaining = [], size
            while remaining:
                block = channel.recv(min(1024 * 1024, remaining))
                if not block:
                    raise EOFError("P2 peer closed early")
                chunks.append(block)
                remaining -= len(block)
            return b"".join(chunks)

        def line(channel):
            value = bytearray()
            while not value.endswith(b"\\n"):
                block = channel.recv(1)
                if not block or len(value) > 65536:
                    raise EOFError("P2 header ended early")
                value.extend(block)
            return bytes(value)

        def signature(header):
            unsigned = {key: value for key, value in header.items() if key != "mac"}
            canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            return hmac.new(credential, canonical, hashlib.sha256).hexdigest()

        def client_echo(host, port, size):
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cert_path)
            context.check_hostname = False
            pattern = b"E025-RUNPOD-NETWORK-CANARY"
            payload = (pattern * ((size // len(pattern)) + 1))[:size]
            nonce = hashlib.sha256(f"{time.time_ns()}:{host}:{port}".encode()).hexdigest()
            header = {"op": "echo", "nonce": nonce, "size": size}
            header["mac"] = signature(header)
            started = time.perf_counter()
            raw = socket.create_connection((host, port), timeout=60)
            with context.wrap_socket(raw, server_hostname=None) as channel:
                connected = time.perf_counter()
                channel.sendall(json.dumps(header, sort_keys=True).encode() + b"\\n" + payload)
                response = json.loads(line(channel))
            elapsed = time.perf_counter() - started
            expected = hmac.new(
                credential,
                f"{nonce}:{size}:{hashlib.sha256(payload).hexdigest()}".encode(),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(str(response.get("mac")), expected):
                raise RuntimeError("P2 response authentication failed")
            return {
                "tls_handshake_seconds": connected - started,
                "round_trip_seconds": elapsed,
                "payload_bytes": size,
                "throughput_mbps": size * 8 / max(elapsed, 1e-9) / 1_000_000,
                "authenticated": True,
            }

        def handle(raw):
            try:
                with server_context.wrap_socket(raw, server_side=True) as channel:
                    header = json.loads(line(channel))
                    if not hmac.compare_digest(str(header.pop("mac", "")), signature(header)):
                        raise PermissionError("P2 request authentication failed")
                    nonce, size = str(header["nonce"]), int(header["size"])
                    if header["op"] == "echo":
                        started = time.perf_counter()
                        payload = exact(channel, size)
                        digest = hashlib.sha256(payload).hexdigest()
                        response = {
                            "received_bytes": len(payload),
                            "sha256": digest,
                            "server_receive_seconds": time.perf_counter() - started,
                        }
                    elif header["op"] == "relay":
                        response = client_echo(str(header["target_host"]), int(header["target_port"]), size)
                        response["received_bytes"] = size
                        response["sha256"] = "relay-verified"
                    else:
                        raise ValueError("unsupported P2 operation")
                    response["mac"] = hmac.new(
                        credential,
                        f"{nonce}:{size}:{response['sha256']}".encode(),
                        hashlib.sha256,
                    ).hexdigest()
                    channel.sendall(json.dumps(response, sort_keys=True).encode() + b"\\n")
            finally:
                raw.close()

        ports = [int(value) for value in os.environ.get("E025_LIGHTWEIGHT_PORTS", "42525").split(",")]
        for port in ports:
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", port))
            listener.listen(32)
            def serve(sock=listener):
                while True:
                    connection, _ = sock.accept()
                    threading.Thread(target=handle, args=(connection,), daemon=True).start()
            threading.Thread(target=serve, daemon=True).start()
        print(json.dumps({"event": "E025_P2_TLS_AUTH_READY", "ports": ports}), flush=True)
        time.sleep(21600)
        """
    )


def runtime_create_payload(
    manifest: dict[str, Any],
    *,
    secrets: dict[str, str] | None,
    fragment_gpu_id: str | None,
    lightweight: bool,
    parent_endpoints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    template = copy.deepcopy(manifest["create_payload_redacted"])
    if manifest["pod_class"] == "FRAGMENT":
        if fragment_gpu_id is None:
            raise RuntimeError("no located validated Ampere fragment GPU is selectable")
        template["gpuTypeIds"] = [fragment_gpu_id]
    else:
        template["gpuTypeIds"] = [BACKBONE_GPU_ID]
    template["gpuCount"] = int(manifest["requested_gpu_count"])
    template["name"] = str(manifest["pod_name"])
    template["imageName"] = IMAGE_TAG
    template["ports"] = [f"{port}/tcp" for port in manifest["internal_serving_ports"]]
    non_secret = dict(manifest["environment"]["non_secret"])
    specs = copy.deepcopy(manifest["worker_specs"])
    if parent_endpoints is not None:
        for spec in specs:
            if str(spec["worker_id"]).endswith("-parent"):
                spec["expert_endpoints"] = parent_endpoints
    non_secret["E025_WORKER_SPECS_B64"] = encode_worker_specs(specs)
    if lightweight:
        template["dockerEntrypoint"] = ["python"]
        network_probe = manifest["pod_class"] == "NETWORK_CANARY"
        template["dockerStartCmd"] = [
            "-c",
            lightweight_start_code(network_probe=network_probe),
        ]
        template["containerDiskInGb"] = 20
        non_secret = {
            "E025_RUN_ID": RUN_ID,
            "E025_RUNPOD_MODE": "LIGHTWEIGHT_ALLOCATION_REHEARSAL_NO_K3",
            "E025_LIGHTWEIGHT_PORTS": ",".join(
                str(port) for port in manifest["internal_serving_ports"]
            ),
        }
        template["env"] = {**non_secret, **(secrets or {})} if network_probe else non_secret
        template["globalNetworking"] = network_probe
    else:
        if secrets is None:
            raise ValueError("paid E025 worker payload requires ephemeral transport secrets")
        template["env"] = {**non_secret, **secrets}
    if len(template["env"]) > 50:
        raise ValueError("RunPod environment variable limit exceeded")
    if template.get("networkVolumeId"):
        raise ValueError("first RunPod E025 attempt forbids network volumes")
    return template


def stage_request_plan(
    *,
    preflight: Path,
    mode: str,
) -> dict[str, Any]:
    try:
        manifests = select_stage_manifests(_manifests(preflight), mode)
    except StopIteration:
        return {
            "schema_version": "experiment-025-runpod-paid-stage-plan-v1",
            "generated_at_utc": utc_now(),
            "mode": mode,
            "status": "DRY_RUN_BLOCKED",
            "provider_mode": "READ_ONLY_PREPARATION",
            "provider_calls": 0,
            "provider_mutations": [],
            "paid_resources_created": 0,
            "watchdog_required_before_first_create": True,
            "rest_create_provider_ttl_field_available": False,
            "independent_watchdog_required": True,
            "fragment_gpu_selected": None,
            "blockers": ["Current preferred topology has no live-supported multi-GPU Pod for P3."],
            "requests": [],
        }
    fragment_gpu = choose_fragment_gpu(preflight)
    requests: list[dict[str, Any]] = []
    blockers: list[str] = []
    for manifest in manifests:
        try:
            payload = runtime_create_payload(
                manifest,
                secrets={
                    "E025_RUN_CREDENTIAL_B64": "<EPHEMERAL>",
                    "E025_TLS_CERT_B64": "<EPHEMERAL>",
                    "E025_TLS_KEY_B64": "<EPHEMERAL>",
                },
                fragment_gpu_id=fragment_gpu,
                lightweight=mode in LIGHTWEIGHT_MODES,
            )
        except RuntimeError as exc:
            blockers.append(str(exc))
            payload = None
        requests.append(
            {
                "logical_pod_id": manifest["logical_pod_id"],
                "pod_class": manifest["pod_class"],
                "payload": redact_provider_payload(payload),
                "payload_build_blocked": payload is None,
            }
        )
    return {
        "schema_version": "experiment-025-runpod-paid-stage-plan-v1",
        "generated_at_utc": utc_now(),
        "mode": mode,
        "status": "DRY_RUN_BLOCKED" if blockers else "DRY_RUN_READY",
        "provider_mode": "READ_ONLY_PREPARATION",
        "provider_calls": 0,
        "provider_mutations": [],
        "paid_resources_created": 0,
        "watchdog_required_before_first_create": True,
        "rest_create_provider_ttl_field_available": False,
        "independent_watchdog_required": True,
        "fragment_gpu_selected": fragment_gpu,
        "blockers": sorted(set(blockers)),
        "requests": requests,
    }


def _wait_for_allocation(
    provider: RunPodProvider,
    pod_id: str,
    *,
    deadline_epoch: float,
    require_public_ports: list[int],
) -> Any:
    last: Any = None
    while time.time() < deadline_epoch:
        value = provider.get_pod(pod_id)
        last = normalize_pod(value)
        terminal = str(last.current_status or last.desired_status or "").upper()
        if terminal in {"EXITED", "TERMINATED", "ERROR"}:
            raise RuntimeError(f"RunPod Pod {pod_id} entered terminal status {terminal}")
        identity_ready = bool(last.machine_id and last.datacenter_id)
        public_endpoint_ready = not require_public_ports or bool(
            last.public_ip and all(port in last.port_mappings for port in require_public_ports)
        )
        if identity_ready and public_endpoint_ready:
            return last
        time.sleep(3.0)
    raise TimeoutError(f"RunPod Pod {pod_id} did not expose identity/ports: {last}")


def _readiness_probe(
    *,
    allocation: Any,
    manifest: dict[str, Any],
    material: dict[str, Any],
    deadline_epoch: float,
) -> list[dict[str, Any]]:
    from .provisioning import _probe_register

    ready: list[dict[str, Any]] = []
    for spec in manifest["worker_specs"]:
        endpoint = resolve_endpoint(
            allocation=allocation,
            worker_id=str(spec["worker_id"]),
            internal_port=int(spec["port"]),
            mode=EndpointMode.RUNPOD_PUBLIC_TCP,
        )
        last_error = "not attempted"
        while time.time() < deadline_epoch:
            try:
                receipt = _probe_register(
                    host=endpoint.host,
                    port=endpoint.port,
                    worker_id=str(spec["worker_id"]),
                    credential=Path(str(material["credential_path"])).read_bytes(),
                    certificate=Path(str(material["certificate_path"])),
                    timeout_seconds=15.0,
                )
                if receipt.get("status") != "READY":
                    raise RuntimeError("worker returned a non-READY receipt")
                ready.append(
                    {
                        "worker_id": spec["worker_id"],
                        "gpu_slot": spec["gpu_slot"],
                        "endpoint": endpoint.authority,
                        "provider_machine_id": allocation.machine_id,
                        "ready": receipt,
                    }
                )
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(5.0)
        else:
            raise TimeoutError(f"worker {spec['worker_id']} was not READY: {last_error}")
    return ready


def _read_json_line(channel: ssl.SSLSocket) -> dict[str, Any]:
    value = bytearray()
    while not value.endswith(b"\n"):
        block = channel.recv(1)
        if not block or len(value) > 65_536:
            raise EOFError("P2 network probe response ended before its JSON line")
        value.extend(block)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("P2 network probe response is not an object")
    return parsed


def _p2_network_request(
    *,
    host: str,
    port: int,
    material: dict[str, Any],
    payload_bytes: int,
    deadline_epoch: float,
    relay_target: tuple[str, int] | None = None,
) -> dict[str, Any]:
    credential = Path(str(material["credential_path"])).read_bytes()
    certificate = Path(str(material["certificate_path"]))
    pattern = b"E025-RUNPOD-NETWORK-CANARY"
    payload = (pattern * ((payload_bytes // len(pattern)) + 1))[:payload_bytes]
    last_error = "not attempted"
    while time.time() < deadline_epoch:
        nonce = hashlib.sha256(
            f"{time.time_ns()}:{host}:{port}:{payload_bytes}".encode()
        ).hexdigest()
        header: dict[str, Any] = {
            "op": "relay" if relay_target else "echo",
            "nonce": nonce,
            "size": payload_bytes,
        }
        if relay_target:
            header["target_host"], header["target_port"] = relay_target
        canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        header["mac"] = hmac.new(credential, canonical, hashlib.sha256).hexdigest()
        try:
            context = ssl.create_default_context(
                ssl.Purpose.SERVER_AUTH,
                cafile=str(certificate),
            )
            context.check_hostname = False
            started = time.perf_counter()
            raw = socket.create_connection((host, port), timeout=60)
            with context.wrap_socket(raw, server_hostname=None) as channel:
                connected = time.perf_counter()
                channel.sendall(json.dumps(header, sort_keys=True).encode() + b"\n")
                if relay_target is None:
                    channel.sendall(payload)
                response = _read_json_line(channel)
            elapsed = time.perf_counter() - started
            digest = (
                "relay-verified"
                if relay_target is not None
                else hashlib.sha256(payload).hexdigest()
            )
            expected = hmac.new(
                credential,
                f"{nonce}:{payload_bytes}:{digest}".encode(),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(str(response.get("mac", "")), expected):
                raise RuntimeError("P2 response HMAC does not match")
            if int(response.get("received_bytes", -1)) != payload_bytes:
                raise RuntimeError("P2 response byte count does not match")
            return {
                "path": "RUNPOD_GLOBAL_PRIVATE" if relay_target else "RUNPOD_PUBLIC_TCP",
                "source_control_endpoint": f"{host}:{port}",
                "relay_target": (f"{relay_target[0]}:{relay_target[1]}" if relay_target else None),
                "payload_bytes": payload_bytes,
                "tls_handshake_seconds": connected - started,
                "round_trip_seconds": elapsed,
                "end_to_end_mbps": payload_bytes * 8 / max(elapsed, 1e-9) / 1_000_000,
                "tls_certificate_pinned": True,
                "hmac_authenticated": True,
                "response": {key: value for key, value in response.items() if key != "mac"},
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5.0)
    raise TimeoutError(f"P2 network probe failed before deadline: {last_error}")


def _run_p2_network_canary(
    *,
    allocations: list[Any],
    material: dict[str, Any],
    deadline_epoch: float,
) -> dict[str, Any]:
    if len(allocations) != 2:
        raise RuntimeError("P2 requires exactly two allocated Secure Pods")
    public = [
        resolve_endpoint(
            allocation=allocation,
            worker_id=f"p2-network-{index}",
            internal_port=42525,
            mode=EndpointMode.RUNPOD_PUBLIC_TCP,
        )
        for index, allocation in enumerate(allocations)
    ]
    rows: list[dict[str, Any]] = []
    for endpoint in public:
        for size in (7_168 * 4, 8 * 1024 * 1024):
            rows.append(
                _p2_network_request(
                    host=endpoint.host,
                    port=endpoint.port,
                    material=material,
                    payload_bytes=size,
                    deadline_epoch=deadline_epoch,
                )
            )
    for source_index, target_index in ((0, 1), (1, 0)):
        source = public[source_index]
        target = allocations[target_index]
        rows.append(
            _p2_network_request(
                host=source.host,
                port=source.port,
                material=material,
                payload_bytes=8 * 1024 * 1024,
                deadline_epoch=deadline_epoch,
                relay_target=(f"{target.pod_id}.runpod.internal", 42525),
            )
        )
    return {
        "schema_version": "experiment-025-runpod-p2-network-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "evidence_class": "PHYSICAL_SHAPED_BY_RUNPOD_NETWORK",
        "controller_location": "LOCAL_WINDOWS",
        "documented_private_network_ceiling_mbps": 100,
        "measurements": rows,
        "selection_must_be_made_from_measurements": True,
    }


def _run_model_stage_validation(
    *,
    repo: Path,
    run_root: Path,
    stage_root: Path,
    mode: str,
    checkpoint: Path,
    manifests: list[dict[str, Any]],
    allocations: list[Any],
    readiness: list[dict[str, Any]],
    material: dict[str, Any],
) -> dict[str, Any]:
    from .canary_runtime import (
        run_physical_stage_fixture,
        run_physical_sub_layer_canary,
    )
    from .controller import run_physical_generation
    from .provisioning import LiveWorker, write_live_endpoints

    ready_by_worker = {str(row["worker_id"]): row for row in readiness}
    workers: list[LiveWorker] = []
    for manifest, allocation in zip(manifests, allocations, strict=True):
        for role, spec in zip(manifest["roles"], manifest["worker_specs"], strict=True):
            worker_id = str(spec["worker_id"])
            ready_row = ready_by_worker[worker_id]
            workers.append(
                LiveWorker(
                    worker_id=worker_id,
                    role=str(role["role"]),
                    layer=int(role["layer"]),
                    worker_index=(
                        None if role.get("worker_index") is None else int(role["worker_index"])
                    ),
                    gpu_slot=int(spec["gpu_slot"]),
                    offer_id=-1,
                    instance_id=allocation.pod_id,  # type: ignore[arg-type]
                    machine_id=allocation.machine_id,  # type: ignore[arg-type]
                    label=str(manifest["pod_name"]),
                    host=str(ready_row["endpoint"]).rsplit(":", maxsplit=1)[0],
                    port=int(str(ready_row["endpoint"]).rsplit(":", maxsplit=1)[1]),
                    gpu_name=str(ready_row["ready"].get("gpu", {}).get("gpu_name", "")),
                    ready=ready_row["ready"],
                )
            )
    if len(workers) != len(readiness):
        raise RuntimeError("RunPod readiness rows do not cover the selected logical roles")
    credential_path = Path(str(material["credential_path"]))
    certificate = Path(str(material["certificate_path"]))
    oracle_root = repo / "artifacts" / "experiment-014" / "oracle-full-93-idot0"
    oracle_trace = oracle_root / "hidden-trace.f32"
    oracle_routes = oracle_root / "routes.txt"
    if mode in {"p1", "p3"}:
        fixtures = [
            run_physical_stage_fixture(
                worker=worker,
                checkpoint=checkpoint,
                oracle_trace=oracle_trace,
                oracle_routes=oracle_routes,
                credential_path=credential_path,
                certificate=certificate,
                output_path=stage_root / f"physical-stage-{worker.layer:03d}.json",
                cycle_id=f"E025-RUNPOD-{mode.upper()}-{worker.layer:03d}",
            )
            for worker in workers
        ]
        gpu_uuids = {
            str(worker.ready.get("gpu", {}).get("gpu_uuid", "unknown")) for worker in workers
        }
        status = (
            "PASS"
            if all(row.get("status") == "PASS" for row in fixtures)
            and len(gpu_uuids) == len(workers)
            and "unknown" not in gpu_uuids
            else "FAIL"
        )
        return {
            "schema_version": f"experiment-025-runpod-{mode}-model-canary-v1",
            "generated_at_utc": utc_now(),
            "status": status,
            "evidence_class": "PHYSICAL",
            "simultaneously_ready_worker_count": len(workers),
            "distinct_gpu_uuid_count": len(gpu_uuids),
            "native_stage_fixtures": fixtures,
            "multi_gpu_physical_validation": mode == "p3",
        }
    if mode == "p4":
        parent = next(worker for worker in workers if worker.role == "SUB_LAYER_PARENT")
        fragments = sorted(
            (worker for worker in workers if worker.role == "SUB_LAYER_WORKER"),
            key=lambda worker: int(worker.worker_index or 0),
        )
        return run_physical_sub_layer_canary(
            parent=parent,
            fragments=fragments,
            checkpoint=checkpoint,
            oracle_trace=oracle_trace,
            oracle_routes=oracle_routes,
            physical_placement=run_root / "preflight" / "physical-placement.json",
            credential_path=credential_path,
            certificate=certificate,
            output_path=stage_root / "runpod-sub-layer-canary.json",
        )
    if mode != "headline":
        raise ValueError(f"no model validation for RunPod stage {mode}")
    endpoint_path = stage_root / "live-endpoints.json"
    write_live_endpoints(endpoint_path, workers)
    correctness = run_physical_generation(
        endpoints_path=endpoint_path,
        credential_path=credential_path,
        certificate=certificate,
        stage_zero_snapshot=checkpoint,
        output_path=stage_root / "physical-two-token.json",
        prompt="Hi",
        max_new_tokens=2,
        fixture_token_ids=[163584, 18699, 11],
        fixture_hidden_trace=oracle_trace,
        fixture_routes=oracle_routes,
    )
    if correctness.get("status") != "PASS":
        raise RuntimeError("RunPod headline two-token physical correctness gate failed")
    budget = read_json(run_root / "preflight" / "runpod" / "headline-token-budget.json")
    generation = run_physical_generation(
        endpoints_path=endpoint_path,
        credential_path=credential_path,
        certificate=certificate,
        stage_zero_snapshot=checkpoint,
        output_path=stage_root / "headline-generation.json",
        prompt=str(budget["prompt"]),
        max_new_tokens=int(budget["generation_max_new_tokens"]),
    )
    exact = str(generation.get("decoded_text", "")).strip() == str(budget["exact_target_text"])
    if generation.get("status") != "PASS" or not exact:
        raise RuntimeError("RunPod headline generation did not reproduce the exact target")
    return {
        "schema_version": "experiment-025-runpod-headline-validation-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "evidence_class": "PHYSICAL",
        "two_token_correctness": correctness,
        "public_generation": generation,
        "exact_target_reproduced": exact,
        "total_consumer_gpus": len(workers),
        "distinct_physical_machine_ids": len({str(worker.machine_id) for worker in workers}),
    }


def execute_paid_stage(
    *,
    repo: Path,
    run_root: Path,
    mode: str,
    allow_paid_run: bool,
    checkpoint: Path = Path(r"F:\models\Kimi-K3"),
) -> dict[str, Any]:
    """Execute a future paid stage; unreachable without both software intents."""

    if mode not in PAID_MODES:
        raise ValueError(f"unsupported paid RunPod mode: {mode}")
    policy = ProviderMutationPolicy.from_intent(allow_paid_run=allow_paid_run)
    policy.require(ProviderOperation.CREATE_POD)
    preflight = run_root / "preflight" / "runpod"
    preparation_status = read_json(preflight / "RUNPOD_PREPARATION_STATUS.json")
    if preparation_status.get("status") != "READY_FOR_PAID_RUNPOD_CANARIES":
        raise RuntimeError(
            "RunPod paid mutation remains blocked by preparation status: "
            f"{preparation_status.get('status')}"
        )
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY must be supplied through the process environment for the paid REST path"
        )
    fragment_gpu = choose_fragment_gpu(preflight)
    if fragment_gpu is None and mode in {"p4", "p5", "headline"}:
        raise RuntimeError("no located validated Ampere fragment GPU is currently selectable")
    manifests = select_stage_manifests(_manifests(preflight), mode)
    stage_root = run_root / "rental" / "runpod" / mode
    stage_root.mkdir(parents=True, exist_ok=False)
    ledger_path = run_root / "rental" / "runpod" / "pod-ledger.jsonl"
    ledger = AppendOnlyRunPodLedger(ledger_path, run_id=RUN_ID)
    deadline_epoch = time.time() + (4 * 3600 if mode == "headline" else 2 * 3600)
    temporary_material = tempfile.TemporaryDirectory(prefix=f"e025-runpod-{mode}-")
    material = create_transport_material(
        Path(temporary_material.name) / "transport",
        RUN_ID,
    )
    secret_values = _secret_values(material)
    try:
        watchdog = start_runpod_watchdog(
            run_id=RUN_ID,
            stage=mode,
            ledger_path=ledger_path,
            deadline_epoch=deadline_epoch,
            receipt_path=stage_root / "watchdog-receipt.json",
            log_path=stage_root / "watchdog-log.jsonl",
            trigger_path=stage_root / "WATCHDOG_TRIGGER",
            stop_path=stage_root / "WATCHDOG_STOP",
            cleanup_receipt_path=stage_root / "watchdog-cleanup.json",
            allow_paid_run=True,
        )
    except BaseException:
        temporary_material.cleanup()
        raise
    if watchdog.get("status") != "RUNNING":
        temporary_material.cleanup()
        raise RuntimeError("independent RunPod watchdog did not start")
    audit: list[dict[str, Any]] = []
    provider = RunPodProvider(
        transport=UrllibRunPodTransport(api_key=api_key, audit=audit.append),
        policy=policy,
    )
    allocations: list[Any] = []
    readiness: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    cleanup: dict[str, Any]
    try:
        fragment_allocations: list[Any] = []
        for manifest in manifests:
            parent_endpoints: list[dict[str, Any]] | None = None
            if manifest["pod_class"] == "LAYER89_PARENT":
                if len(fragment_allocations) != 4:
                    raise RuntimeError("parent creation requires four allocated fragments")
                if len({row.machine_id for row in fragment_allocations}) != 4:
                    raise RuntimeError("fragment machineId values are not pairwise distinct")
                parent_endpoints = [
                    {
                        "worker_id": f"e025-layer-089-sub-{index:02d}",
                        "worker_index": index,
                        "host": resolve_endpoint(
                            allocation=allocation,
                            worker_id=f"e025-layer-089-sub-{index:02d}",
                            internal_port=42525,
                            mode=EndpointMode.RUNPOD_PUBLIC_TCP,
                            fragment_endpoint_generation=1,
                        ).host,
                        "port": resolve_endpoint(
                            allocation=allocation,
                            worker_id=f"e025-layer-089-sub-{index:02d}",
                            internal_port=42525,
                            mode=EndpointMode.RUNPOD_PUBLIC_TCP,
                            fragment_endpoint_generation=1,
                        ).port,
                        "timeout_seconds": 180.0,
                        "fragment_endpoint_generation": 1,
                    }
                    for index, allocation in enumerate(fragment_allocations)
                ]
            payload = runtime_create_payload(
                manifest,
                secrets=secret_values,
                fragment_gpu_id=fragment_gpu,
                lightweight=mode in LIGHTWEIGHT_MODES,
                parent_endpoints=parent_endpoints,
            )
            allocation = None
            for allocation_attempt in range(1, 5):
                response = provider.create_pod(payload)
                candidate = normalize_pod(response)
                if not candidate.pod_id:
                    raise RuntimeError("RunPod creation response has no Pod ID")
                # The first action after a successful create response is append-only
                # attribution; no readiness or endpoint step can precede it.
                ledger.record_created(
                    pod_id=candidate.pod_id,
                    pod_name=str(manifest["pod_name"]),
                    stage=mode,
                )
                requested_ports = [int(value) for value in manifest["internal_serving_ports"]]
                candidate = _wait_for_allocation(
                    provider,
                    candidate.pod_id,
                    deadline_epoch=deadline_epoch,
                    # A fragment's authoritative machineId is enough to reject
                    # duplicate placement. Do that before waiting for network
                    # mappings or model readiness so wasted acquisition is kept
                    # as short as the provider identity publication permits.
                    require_public_ports=(
                        [] if manifest["pod_class"] == "FRAGMENT" else requested_ports
                    ),
                )
                duplicate_fragment_machine = manifest[
                    "pod_class"
                ] == "FRAGMENT" and candidate.machine_id in {
                    row.machine_id for row in fragment_allocations
                }
                if not duplicate_fragment_machine:
                    if manifest["pod_class"] == "FRAGMENT":
                        candidate = _wait_for_allocation(
                            provider,
                            candidate.pod_id,
                            deadline_epoch=deadline_epoch,
                            require_public_ports=requested_ports,
                        )
                    allocation = candidate
                    break
                # Detect and reject at provider identity publication, before waiting
                # for worker READY/model acquisition completion.
                provider.delete_pod(candidate.pod_id)
                ledger.append(
                    "POD_DELETE_REQUESTED",
                    pod_id=candidate.pod_id,
                    pod_name=manifest["pod_name"],
                    reason="duplicate-fragment-machineId",
                    permanent_delete=True,
                    replacement_attempt=allocation_attempt,
                )
            if allocation is None:
                raise RuntimeError(
                    "four attempts could not place a fragment on a distinct machineId"
                )
            if manifest["pod_class"] == "FRAGMENT":
                fragment_allocations.append(allocation)
            allocations.append(allocation)
        # All provider identities are established before waiting for expensive
        # model readiness. In particular, the parent starts as soon as the four
        # fragment endpoint identities exist, while every fragment is still free
        # to acquire/load concurrently.
        if mode not in LIGHTWEIGHT_MODES:
            for manifest, allocation in zip(manifests, allocations, strict=True):
                readiness.extend(
                    _readiness_probe(
                        allocation=allocation,
                        manifest=manifest,
                        material=material,
                        deadline_epoch=deadline_epoch,
                    )
                )
        if mode == "p5" and len(allocations) != len(manifests):
            raise RuntimeError("P5 did not allocate the complete intended topology")
        stage_validation: dict[str, Any] | None = None
        if mode == "p2":
            stage_validation = _run_p2_network_canary(
                allocations=allocations,
                material=material,
                deadline_epoch=deadline_epoch,
            )
            atomic_write_json(stage_root / "network-canary.json", stage_validation)
        elif mode in {"p1", "p3", "p4", "headline"}:
            stage_validation = _run_model_stage_validation(
                repo=repo,
                run_root=run_root,
                stage_root=stage_root,
                mode=mode,
                checkpoint=checkpoint.resolve(),
                manifests=manifests,
                allocations=allocations,
                readiness=readiness,
                material=material,
            )
            if stage_validation.get("status") != "PASS":
                raise RuntimeError(f"RunPod {mode} physical validation failed")
            atomic_write_json(
                stage_root / f"{mode}-validation.json",
                stage_validation,
            )
        elif mode == "p5":
            time.sleep(30.0)
            refreshed = [
                normalize_pod(provider.get_pod(allocation.pod_id)) for allocation in allocations
            ]
            fragment_machines = {
                allocation.machine_id
                for manifest, allocation in zip(manifests, refreshed, strict=True)
                if manifest["pod_class"] == "FRAGMENT"
            }
            stable_identities = all(
                before.machine_id == after.machine_id
                and before.datacenter_id == after.datacenter_id
                and before.public_ip == after.public_ip
                for before, after in zip(allocations, refreshed, strict=True)
            )
            if len(fragment_machines) != 4 or None in fragment_machines:
                raise RuntimeError("P5 lost four distinct fragment machineId values")
            if not stable_identities:
                raise RuntimeError("P5 Pod identities changed during the barrier")
            stage_validation = {
                "schema_version": "experiment-025-runpod-p5-acquisition-barrier-v1",
                "generated_at_utc": utc_now(),
                "status": "PASS",
                "evidence_class": "PHYSICAL_PROVIDER_CONTROL_PLANE_ONLY",
                "k3_download_performed": False,
                "barrier_seconds": 30,
                "planned_pod_count": len(manifests),
                "allocated_pod_count": len(refreshed),
                "logical_gpu_roles": sum(
                    int(manifest["requested_gpu_count"]) for manifest in manifests
                ),
                "fragment_machine_ids_distinct": True,
                "network_identities_stable": stable_identities,
            }
            atomic_write_json(
                stage_root / "full-acquisition-rehearsal.json",
                stage_validation,
            )
        receipt = {
            "schema_version": "experiment-025-runpod-paid-stage-v1",
            "generated_at_utc": utc_now(),
            "status": "PASS",
            "evidence_class": (
                "PHYSICAL_PROVIDER_CONTROL_PLANE_ONLY"
                if mode in LIGHTWEIGHT_MODES
                else "PHYSICAL_RUNPOD_CANARY"
            ),
            "mode": mode,
            "allocations": [allocation_dict(row) for row in allocations],
            "readiness": readiness,
            "stage_validation": stage_validation,
            "watchdog": watchdog,
            "request_audit": audit,
        }
        atomic_write_json(stage_root / "stage-receipt.json", receipt)
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc)[:2000]}
        raise
    finally:
        try:
            cleanup = cleanup_from_ledger(
                provider=provider,
                ledger_path=ledger_path,
                run_id=RUN_ID,
                reason=f"paid-stage:{mode}:finally",
            )
            atomic_write_json(stage_root / "cleanup-receipt.json", cleanup)
            if cleanup["zero_live_attributed_pods"]:
                (stage_root / "WATCHDOG_STOP").write_text("cleanup verified\n", encoding="utf-8")
            if failure is not None:
                atomic_write_json(stage_root / "failure.json", failure)
        finally:
            temporary_material.cleanup()
    return receipt


__all__ = [
    "LIGHTWEIGHT_MODES",
    "PAID_MODES",
    "choose_fragment_gpu",
    "execute_paid_stage",
    "lightweight_start_code",
    "runtime_create_payload",
    "select_stage_manifests",
    "stage_request_plan",
]
