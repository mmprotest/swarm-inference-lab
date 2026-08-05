"""Experiment 010 compatibility imports for canonical Colibri expert banks."""

from swarm_inference.backends.colibri.expert_bank import *  # noqa: F403
from swarm_inference.backends.colibri.expert_bank import _finish_target, main  # noqa: F401

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
