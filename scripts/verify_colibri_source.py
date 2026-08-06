"""Bounded source-tree preflight for the pinned Colibri submodule contract."""

from __future__ import annotations

import json
from pathlib import Path

from swarm_inference.backends.colibri.dependency import verify_colibri_source_contract


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    result = verify_colibri_source_contract(repository)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
