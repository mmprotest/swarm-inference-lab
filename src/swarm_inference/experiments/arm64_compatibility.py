"""ARM64 cross-build and QEMU protocol compatibility evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.backend_environments import run_logged
from swarm_inference.protocol.checksums import sha256_file


def build_and_test_arm64(
    *,
    repository_root: Path,
    backend_root: Path,
    llamacpp_environment: dict[str, Any],
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(str(llamacpp_environment["source_path"])).resolve()
    output = (
        backend_root / "llamacpp" / f"build-arm64-{str(llamacpp_environment['source_commit'])[:12]}"
    )
    probe_source = repository_root / "native" / "experiment_007_arm64_worker.cpp"
    probe_binary = output / "bin" / "universal-worker-probe"
    logs = run_directory / "logs"
    output.mkdir(parents=True, exist_ok=True)
    image = "ubuntu:24.04"
    build_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{source}:/llama:ro",
        "-v",
        f"{output.resolve()}:/build",
        "-v",
        f"{probe_source.resolve()}:/probe.cpp:ro",
        image,
        "bash",
        "-lc",
        (
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "cmake ninja-build g++ gcc-aarch64-linux-gnu g++-aarch64-linux-gnu && "
            "cmake -S /llama -B /build/llama -G Ninja "
            "-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 "
            "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc "
            "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ "
            "-DHOST_CXX_COMPILER=/usr/bin/g++ "
            "-DCMAKE_EXE_LINKER_FLAGS='-static-libgcc -static-libstdc++' "
            "-DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF "
            "-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON && "
            "cmake --build /build/llama --target llama-server -j2 && "
            "mkdir -p /build/bin && cp /build/llama/bin/llama-server /build/bin/llama-server && "
            "aarch64-linux-gnu-g++ -std=c++17 -O2 -static /probe.cpp "
            "-o /build/bin/universal-worker-probe"
        ),
    ]
    if not probe_binary.is_file() or not (output / "bin" / "llama-server").is_file():
        build = run_logged(
            build_command,
            cwd=repository_root,
            log_root=logs,
            name="arm64-cross-build",
            timeout_seconds=7200,
        )
    else:
        build = None
    server = output / "bin" / "llama-server"
    build_pass = (
        probe_binary.is_file() and server.is_file() and (build is None or build.return_code == 0)
    )
    build_evidence = {
        "classification": "arm64_compatibility",
        "status": "PASS" if build_pass else "FAIL",
        "target": "aarch64-linux-gnu",
        "source_commit": llamacpp_environment["source_commit"],
        "compiler": "aarch64-linux-gnu-g++",
        "build_command": build_command,
        "llamacpp_worker_path": str(server),
        "llamacpp_worker_sha256": sha256_file(server) if server.is_file() else None,
        "abi_client_path": str(probe_binary),
        "abi_client_sha256": sha256_file(probe_binary) if probe_binary.is_file() else None,
        "raspberry_pi_performance": "unproven",
    }
    if not build_pass:
        return build_evidence, {
            "classification": "arm64_compatibility",
            "status": "FAIL",
            "diagnostic": "cross-build did not produce both binaries",
            "raspberry_pi_performance": "unproven",
        }
    qemu_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{output.resolve()}:/build:ro",
        image,
        "bash",
        "-lc",
        (
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y qemu-user-static && "
            "qemu-aarch64-static /build/bin/universal-worker-probe "
            "--expected-shard-hash 007abc --actual-shard-hash 007abc --cancel"
        ),
    ]
    qemu = run_logged(
        qemu_command,
        cwd=repository_root,
        log_root=logs,
        name="arm64-qemu-protocol",
        timeout_seconds=1800,
    )
    payload: dict[str, Any] = {}
    if qemu.return_code == 0:
        for line in reversed(Path(qemu.stdout_path).read_text(encoding="utf-8").splitlines()):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break
    required = (
        payload.get("registration") == "accepted"
        and payload.get("shard_hash_valid") is True
        and payload.get("heartbeat") == "ok"
        and payload.get("cancellation") == "cancelled"
        and payload.get("shutdown") == "clean"
        and isinstance(payload.get("deterministic_output_token"), int)
    )
    protocol = {
        "classification": "arm64_compatibility",
        "status": "PASS" if qemu.return_code == 0 and required else "FAIL",
        "qemu_command": qemu_command,
        "probe": payload,
        "registration": payload.get("registration"),
        "capability_report": payload.get("capabilities"),
        "shard_hash_validation": payload.get("shard_hash_valid"),
        "token_protocol": "PASS"
        if isinstance(payload.get("deterministic_output_token"), int)
        else "FAIL",
        "cancellation": payload.get("cancellation"),
        "heartbeat": payload.get("heartbeat"),
        "clean_shutdown": payload.get("shutdown"),
        "arm64_protocol_compatibility": "measured",
        "qemu_speed_in_contribution_calculations": False,
        "raspberry_pi_performance": "unproven",
    }
    return build_evidence, protocol
