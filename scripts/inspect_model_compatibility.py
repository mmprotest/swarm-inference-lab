"""Inspect current Hugging Face/local checkpoint metadata without acquiring weights."""

from __future__ import annotations

import argparse
import json

from swarm_inference.model.discovery import discover_models


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve immutable model metadata and report whether an installed architecture "
            "adapter recognizes each checkpoint. This does not validate execution."
        )
    )
    parser.add_argument("models", nargs="+", help="Hugging Face IDs/URLs or local paths")
    parser.add_argument("--maximum-models", type=int, default=32)
    arguments = parser.parse_args()
    report = discover_models(arguments.models, maximum_models=arguments.maximum_models)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if all(item.status == "PROFILED" for item in report.records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
