"""Strict read-only loading of the frozen Experiment 022 inventory suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import canonical_sha256
from swarm_inference.experiments.experiment_022.models import Inventory

FROZEN_PLANNER_SETTINGS_SHA256 = (
    "a7a7d65b9687c32307ec517d661b3d748457fd8603adc60b0c9607256045dc8a"
)
FROZEN_THRESHOLDS_SHA256 = (
    "7f13d084f59c55eafd78404ff410e6569b2bd9ec66f11204af1e35e67ca782f9"
)
FROZEN_OUTCOME_CATEGORIES_SHA256 = (
    "647d4a4a9c2aa0b8a34b0b1be5186afc187898f3208ffe2315ffbc97aaee2976"
)


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_frozen_inventories(
    repo: Path,
    freeze_path: Path | None = None,
) -> tuple[list[Inventory], dict[str, Any]]:
    """Load exactly the hashed preregistered suite without regeneration."""

    root = repo.resolve()
    freeze = (
        freeze_path.resolve()
        if freeze_path is not None
        else root
        / "artifacts"
        / "experiment-022"
        / "completion"
        / "frozen-inputs.json"
    )
    receipt = _read(freeze)
    if not bool(receipt.get("frozen_inputs_recoverable")):
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE")
    entries = list(receipt.get("inventories", ()))
    if int(receipt.get("inventory_count", -1)) != 27 or len(entries) != 27:
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE: inventory count is not 27")
    frozen_value_hashes = {
        "planner_settings": _canonical_value_sha256(
            receipt.get("planner_settings")
        ),
        "thresholds": _canonical_value_sha256(receipt.get("thresholds")),
        "outcome_categories": _canonical_value_sha256(
            receipt.get("outcome_categories")
        ),
    }
    expected_value_hashes = {
        "planner_settings": FROZEN_PLANNER_SETTINGS_SHA256,
        "thresholds": FROZEN_THRESHOLDS_SHA256,
        "outcome_categories": FROZEN_OUTCOME_CATEGORIES_SHA256,
    }
    if frozen_value_hashes != expected_value_hashes:
        raise RuntimeError(
            "E022_FROZEN_INPUTS_UNRECOVERABLE: planner settings, thresholds, "
            "or outcome categories changed"
        )

    immutable_core: list[dict[str, Any]] = []
    for entry in receipt.get("core_files", ()):
        relative = str(entry.get("path", "")).replace("\\", "/")
        # Completion implementation necessarily changes the E022 source.  The
        # preregistered artifacts themselves are immutable and are rehashed on
        # every load rather than trusted because a freeze receipt once existed.
        immutable_source = relative.endswith(
            "/experiment_022/models.py"
        ) or relative.endswith("/experiment_022/finalize.py")
        if (
            not relative.startswith("artifacts/experiment-022/")
            and not immutable_source
        ):
            continue
        path = root / relative
        actual = _raw_sha256(path)
        if actual != str(entry.get("sha256", "")):
            raise RuntimeError(
                f"E022_FROZEN_INPUTS_UNRECOVERABLE: frozen core hash changed for {path}"
            )
        immutable_core.append({"path": relative, "sha256": actual})

    immutable_diagnostics: list[dict[str, Any]] = []
    for entry in receipt.get("original_diagnostics", ()):
        relative = str(entry.get("path", "")).replace("\\", "/")
        path = root / relative
        actual = _raw_sha256(path)
        if actual != str(entry.get("sha256", "")):
            raise RuntimeError(
                "E022_FROZEN_INPUTS_UNRECOVERABLE: original failed-run "
                f"diagnostic changed for {path}"
            )
        immutable_diagnostics.append({"path": relative, "sha256": actual})

    immutable_selected_manifests: list[dict[str, Any]] = []
    selected = list(receipt.get("selected_correctness_manifests", ()))
    if len(selected) != 5:
        raise RuntimeError(
            "E022_FROZEN_INPUTS_UNRECOVERABLE: representative selection count changed"
        )
    for entry in selected:
        relative = str(entry.get("path", "")).replace("\\", "/")
        path = root / relative
        actual = _raw_sha256(path)
        if actual != str(entry.get("sha256", "")):
            raise RuntimeError(
                "E022_FROZEN_INPUTS_UNRECOVERABLE: selected representative "
                f"manifest changed for {path}"
            )
        immutable_selected_manifests.append(
            {
                "selection_id": str(entry.get("selection_id", "")),
                "path": relative,
                "sha256": actual,
            }
        )

    suite_entries = [
        row
        for row in receipt.get("core_files", ())
        if str(row.get("path", "")).replace("\\", "/").endswith(
            "/inventories/inventory-suite.json"
        )
    ]
    if len(suite_entries) != 1:
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE: suite receipt is absent")
    suite_entry = suite_entries[0]
    suite_path = root / str(suite_entry["path"])
    if _raw_sha256(suite_path) != str(suite_entry["sha256"]):
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE: suite hash changed")
    suite = _read(suite_path)
    if int(suite.get("inventory_count", -1)) != 27:
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE: suite count changed")
    suite_rows = {
        str(row["inventory_id"]): row for row in suite.get("inventories", ())
    }
    if len(suite_rows) != 27:
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE: suite IDs are ambiguous")

    inventories: list[Inventory] = []
    audited: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        path = root / str(entry["path"])
        raw_hash = _raw_sha256(path)
        if raw_hash != str(entry["sha256"]):
            raise RuntimeError(f"E022_FROZEN_INPUTS_UNRECOVERABLE: hash changed for {path}")
        value = _read(path)
        embedded = str(value.get("inventory_sha256", ""))
        canonical_input = dict(value)
        canonical_input.pop("inventory_sha256", None)
        canonical = canonical_sha256(canonical_input)
        if canonical != embedded or canonical != str(entry["embedded_canonical_sha256"]):
            raise RuntimeError(
                f"E022_FROZEN_INPUTS_UNRECOVERABLE: canonical hash changed for {path}"
            )
        inventory_id = str(value.get("inventory_id", ""))
        seed = int(value.get("seed", -1))
        if inventory_id != str(entry["inventory_id"]) or seed != int(entry["seed"]):
            raise RuntimeError(
                f"E022_FROZEN_INPUTS_UNRECOVERABLE: ID/seed changed for {path}"
            )
        if inventory_id in seen or inventory_id not in suite_rows:
            raise RuntimeError(
                f"E022_FROZEN_INPUTS_UNRECOVERABLE: duplicate/unknown ID {inventory_id}"
            )
        suite_row = suite_rows[inventory_id]
        if (
            int(suite_row["seed"]) != seed
            or str(suite_row["inventory_sha256"]) != canonical
        ):
            raise RuntimeError(
                f"E022_FROZEN_INPUTS_UNRECOVERABLE: suite mismatch for {inventory_id}"
            )
        seen.add(inventory_id)
        inventories.append(Inventory.from_dict(value))
        audited.append(
            {
                "inventory_id": inventory_id,
                "seed": seed,
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "raw_sha256": raw_hash,
                "canonical_sha256": canonical,
            }
        )

    if seen != set(suite_rows):
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE: suite membership changed")
    recorded_seed_set = {
        int(seed)
        for family in receipt.get("seeds", {}).values()
        for seed in family
    }
    inventory_seed_set = {value.seed for value in inventories}
    if recorded_seed_set != inventory_seed_set or len(recorded_seed_set) != 27:
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE: frozen seeds changed")
    audit = {
        "status": "PASS",
        "inventory_count": len(inventories),
        "inventory_ids": [value.inventory_id for value in inventories],
        "seeds": [value.seed for value in inventories],
        "suite_raw_sha256": _raw_sha256(suite_path),
        "suite_canonical_sha256": str(receipt["inventory_suite_sha256"]),
        "regenerated": False,
        "immutable_core_files": immutable_core,
        "immutable_original_diagnostics": immutable_diagnostics,
        "immutable_selected_manifests": immutable_selected_manifests,
        "frozen_value_hashes": frozen_value_hashes,
        "inventories": audited,
    }
    return inventories, audit


__all__ = ["load_frozen_inventories"]
