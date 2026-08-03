from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from swarm_inference.experiments.experiment_010.colibri_expert_bank import (
    _finish_target,
    build_coordinator_container,
    build_expert_bank,
    build_microshard_bank,
    require_bank_ownership,
    verify_bank,
    verify_coordinator_container,
    verify_microshard_reconstruction,
)
from swarm_inference.experiments.experiment_010.colibri_token_path import (
    audit_capacity_ownership,
)

HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 3


def test_bank_publish_retries_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / ".bank.building"
    target = tmp_path / "bank"
    stage.mkdir()
    original = Path.replace
    attempts = 0

    def transient_replace(path: Path, destination: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "transient sharing violation", str(path))
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", transient_replace)
    assert _finish_target(stage, target) == target
    assert target.is_dir()
    assert attempts == 2


def _expert_values(offset: int) -> tuple[np.ndarray, np.ndarray]:
    count = 3 * HIDDEN_SIZE * INTERMEDIATE_SIZE
    weights = (np.arange(count, dtype=np.int16) + offset).astype(np.int8)
    scales = np.linspace(0.01 + offset / 1000, 0.1 + offset / 1000, 10, dtype=np.float32)
    return weights, scales


@pytest.fixture
def native_model(tmp_path: Path) -> tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]]:
    root = tmp_path / "model"
    root.mkdir()
    config = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": 2,
        "num_experts": 3,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    values = {expert_id: _expert_values(17 * expert_id) for expert_id in (0, 1)}
    tensors: dict[str, np.ndarray] = {}
    tensors["model.embed_tokens.weight"] = np.arange(16, dtype=np.float32).reshape(4, 4)
    for expert_id, (weights, scales) in values.items():
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        tensors[f"{prefix}.merged_weight"] = weights
        tensors[f"{prefix}.qs"] = scales
    save_file(tensors, root / "model-00000.safetensors", metadata={"format": "pt"})
    return root, values


def test_capacity_coordinator_owns_no_routed_experts(
    native_model: tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]], tmp_path: Path
) -> None:
    source, _ = native_model
    coordinator = build_coordinator_container(
        source,
        tmp_path / "coordinator-model",
        creation_timestamp="2026-08-02T00:00:00Z",
    )
    result = verify_coordinator_container(coordinator)
    assert result["valid"] is True
    assert result["coordinator_owned_routed_expert_count"] == 0
    assert result["coordinator_owned_routed_expert_bytes"] == 0
    tensors = _read_tensors(coordinator / "model.safetensors")
    assert set(tensors) == {"model.embed_tokens.weight"}
    source_tensors = _read_tensors(source / "model-00000.safetensors")
    assert (
        tensors["model.embed_tokens.weight"].tobytes()
        == source_tensors["model.embed_tokens.weight"].tobytes()
    )


def test_capacity_each_worker_under_ownership_limit(tmp_path: Path) -> None:
    source = tmp_path / "capacity-source"
    source.mkdir()
    config = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": 1,
        "num_experts": 4,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.arange(16, dtype=np.float32).reshape(4, 4)
    }
    for expert_id in range(4):
        weights, scales = _expert_values(17 * expert_id)
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        tensors[f"{prefix}.merged_weight"] = weights
        tensors[f"{prefix}.qs"] = scales
    save_file(tensors, source / "model.safetensors", metadata={"format": "pt"})
    coordinator = build_coordinator_container(source, tmp_path / "coordinator")
    banks = [
        build_expert_bank(
            source,
            tmp_path / f"worker-{expert_id}",
            worker_id=f"worker-{expert_id}",
            owned_experts=[(0, expert_id)],
        )
        for expert_id in range(4)
    ]
    audit = audit_capacity_ownership(
        coordinator_model=coordinator,
        bank_paths=banks,
    )
    assert audit["valid"] is True
    assert audit["globally_owned_expert_count"] == 4
    assert not audit["missing_experts"]
    assert not audit["duplicate_ownership"]
    assert all(row["ownership_fraction"] == 0.25 for row in audit["workers"])
    assert all(row["under_30_percent"] for row in audit["workers"])


def _read_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as handle:
        names = handle.keys()
        return {name: handle.get_tensor(name).copy() for name in names}


def test_native_expert_bank_byte_identity(
    native_model: tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]], tmp_path: Path
) -> None:
    source, values = native_model
    bank = build_expert_bank(
        source,
        tmp_path / "worker-bank",
        worker_id="worker-a",
        owned_experts=[(0, 1)],
        creation_timestamp="2026-08-02T00:00:00Z",
    )

    result = verify_bank(bank)
    assert result["valid"] is True
    assert result["tensor_count"] == 2
    tensors = _read_tensors(bank / "experts.safetensors")
    prefix = "model.layers.0.mlp.experts.1"
    assert set(tensors) == {f"{prefix}.merged_weight", f"{prefix}.qs"}
    assert tensors[f"{prefix}.merged_weight"].tobytes() == values[1][0].tobytes()
    assert tensors[f"{prefix}.qs"].tobytes() == values[1][1].tobytes()

    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    for name, tensor in tensors.items():
        digest = "sha256:" + hashlib.sha256(tensor.tobytes()).hexdigest()
        assert manifest["source_tensor_hashes"][name] == digest
        assert manifest["destination_tensor_hashes"][name] == digest
    assert manifest["total_expert_bytes"] == values[1][0].nbytes + values[1][1].nbytes


def test_native_expert_bank_rejects_unowned_expert(
    native_model: tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]], tmp_path: Path
) -> None:
    source, _ = native_model
    bank = build_expert_bank(
        source,
        tmp_path / "worker-bank",
        worker_id="worker-a",
        owned_experts=[(0, 1)],
        source_model_fingerprint="sha256:" + "7" * 64,
    )

    require_bank_ownership(bank, layer_id=0, expert_id=1)
    with pytest.raises(PermissionError, match="forbids unowned"):
        require_bank_ownership(bank, layer_id=0, expert_id=0)
    with pytest.raises(PermissionError, match="forbids unowned"):
        require_bank_ownership(bank, layer_id=1, expert_id=1)


def _expected_slice(
    weights: np.ndarray, scales: np.ndarray, start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    matrix_values = HIDDEN_SIZE * INTERMEDIATE_SIZE
    gate = weights[:matrix_values].reshape(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    up = weights[matrix_values : 2 * matrix_values].reshape(
        INTERMEDIATE_SIZE, HIDDEN_SIZE
    )
    down = weights[2 * matrix_values :].reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    selected_weights = np.concatenate(
        (gate[start:end].ravel(), up[start:end].ravel(), down[:, start:end].ravel())
    )
    selected_scales = np.concatenate(
        (
            scales[start:end],
            scales[INTERMEDIATE_SIZE + start : INTERMEDIATE_SIZE + end],
            scales[2 * INTERMEDIATE_SIZE :],
        )
    )
    return selected_weights, selected_scales


def test_native_microshard_bank_byte_identity(
    native_model: tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]], tmp_path: Path
) -> None:
    source, values = native_model
    bank = build_microshard_bank(
        source,
        tmp_path / "worker-microshard-bank",
        worker_id="worker-slice-a",
        owned_microshards=[(0, 1, 1, 3)],
        layout="asymmetric",
        source_model_fingerprint="sha256:" + "8" * 64,
    )

    result = verify_bank(bank)
    assert result["valid"] is True
    tensors = _read_tensors(bank / "shards.safetensors")
    prefix = "model.layers.0.mlp.experts.1.microshards.1_3"
    expected_weights, expected_scales = _expected_slice(*values[1], 1, 3)
    assert tensors[f"{prefix}.merged_weight"].tobytes() == expected_weights.tobytes()
    assert tensors[f"{prefix}.qs"].tobytes() == expected_scales.tobytes()
    require_bank_ownership(
        bank, layer_id=0, expert_id=1, hidden_start=1, hidden_end=3
    )
    with pytest.raises(PermissionError, match="forbids unowned"):
        require_bank_ownership(
            bank, layer_id=0, expert_id=1, hidden_start=0, hidden_end=1
        )


def test_native_microshard_reconstruction(
    native_model: tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]], tmp_path: Path
) -> None:
    source, values = native_model
    ranges = ((0, 1), (1, 3))
    reconstructed_gate = np.empty((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=np.int8)
    reconstructed_up = np.empty_like(reconstructed_gate)
    reconstructed_down = np.empty((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=np.int8)
    reconstructed_gate_scales = np.empty(INTERMEDIATE_SIZE, dtype=np.float32)
    reconstructed_up_scales = np.empty(INTERMEDIATE_SIZE, dtype=np.float32)
    observed_down_scales: list[np.ndarray] = []
    banks: list[Path] = []

    for index, (start, end) in enumerate(ranges):
        bank = build_microshard_bank(
            source,
            tmp_path / f"worker-microshard-{index}",
            worker_id=f"worker-{index}",
            owned_microshards=[(0, 1, start, end)],
            layout="equal",
            source_model_fingerprint="sha256:" + "9" * 64,
        )
        banks.append(bank)
        tensors = _read_tensors(bank / "shards.safetensors")
        prefix = f"model.layers.0.mlp.experts.1.microshards.{start}_{end}"
        weights = tensors[f"{prefix}.merged_weight"]
        scales = tensors[f"{prefix}.qs"]
        width = end - start
        gate_count = width * HIDDEN_SIZE
        reconstructed_gate[start:end] = weights[:gate_count].reshape(width, HIDDEN_SIZE)
        reconstructed_up[start:end] = weights[gate_count : 2 * gate_count].reshape(
            width, HIDDEN_SIZE
        )
        reconstructed_down[:, start:end] = weights[2 * gate_count :].reshape(
            HIDDEN_SIZE, width
        )
        reconstructed_gate_scales[start:end] = scales[:width]
        reconstructed_up_scales[start:end] = scales[width : 2 * width]
        observed_down_scales.append(scales[2 * width :])

    source_weights, source_scales = values[1]
    matrix_values = HIDDEN_SIZE * INTERMEDIATE_SIZE
    assert reconstructed_gate.tobytes() == source_weights[:matrix_values].tobytes()
    assert reconstructed_up.tobytes() == source_weights[
        matrix_values : 2 * matrix_values
    ].tobytes()
    assert reconstructed_down.tobytes() == source_weights[2 * matrix_values :].tobytes()
    assert reconstructed_gate_scales.tobytes() == source_scales[:INTERMEDIATE_SIZE].tobytes()
    assert reconstructed_up_scales.tobytes() == source_scales[
        INTERMEDIATE_SIZE : 2 * INTERMEDIATE_SIZE
    ].tobytes()
    for down_scales in observed_down_scales:
        assert down_scales.tobytes() == source_scales[2 * INTERMEDIATE_SIZE :].tobytes()
    verification = verify_microshard_reconstruction(
        source, banks, layer_id=0, expert_id=1
    )
    assert verification["valid"] is True
    assert verification["ranges"] == [[0, 1], [1, 3]]
