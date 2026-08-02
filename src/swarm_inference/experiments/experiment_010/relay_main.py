"""Subprocess entry point for the Experiment 010 relay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_010.relay import RelayServer
from swarm_inference.experiments.experiment_010.schemas import NetworkShapeProfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    target_host, _, target_port = str(configuration["target_endpoint"]).rpartition(":")
    server = RelayServer(
        (str(configuration["host"]), int(configuration["port"])),
        (target_host, int(target_port)),
        NetworkShapeProfile.model_validate(configuration["profile"]),
        Path(configuration["metrics_path"]),
    )
    endpoint = f"{server.server_address[0]}:{server.server_address[1]}"
    arguments.ready.write_text(
        json.dumps({"endpoint": endpoint, "pid": __import__("os").getpid()}) + "\n",
        encoding="utf-8",
    )
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.save_metrics()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
