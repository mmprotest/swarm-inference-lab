"""Run the actual framed protocol with E021's modeled network shape."""

from __future__ import annotations

import json
from pathlib import Path

from swarm_inference.experiments.experiment_020.network_rehearsal import (
    run_network_shaped_rehearsal,
)
from swarm_inference.experiments.experiment_020.vast import atomic_write_json


def main() -> int:
    result = run_network_shaped_rehearsal()
    output = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "experiment-020"
        / "runtime"
        / "network-shaped-rehearsal.json"
    )
    atomic_write_json(output, result)
    print(json.dumps({"status": result["status"], "messages": result["messages"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
