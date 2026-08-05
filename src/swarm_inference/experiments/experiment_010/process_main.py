"""Deprecated entry-point compatibility for the frozen Experiment 010 worker."""

from swarm_inference.experiments.experiment_010.legacy_runtime.process_main import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
