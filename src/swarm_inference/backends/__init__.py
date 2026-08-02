"""Concrete and interface-only Universal Worker backend adapters."""

from swarm_inference.backends.colibri import ColibriBackend
from swarm_inference.backends.interfaces import InterfaceOnlyAdapter
from swarm_inference.backends.llamacpp import LlamaCppAdapter
from swarm_inference.backends.sglang import SGLangAdapter
from swarm_inference.backends.torch_rank import TorchRankAdapter

__all__ = [
    "ColibriBackend",
    "InterfaceOnlyAdapter",
    "LlamaCppAdapter",
    "SGLangAdapter",
    "TorchRankAdapter",
]
