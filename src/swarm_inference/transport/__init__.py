"""Transport abstraction and gRPC AsyncIO implementation."""

from typing import TYPE_CHECKING, Any

from .base import ActivationTransport

if TYPE_CHECKING:
    from .grpc_transport import GrpcTransport, WorkerRpcServer
    from .stage_ring_connection import StageRingConnectionPool
    from .stage_ring_server import StageRingServer

__all__ = [
    "ActivationTransport",
    "GrpcTransport",
    "StageRingConnectionPool",
    "StageRingServer",
    "WorkerRpcServer",
]


def __getattr__(name: str) -> Any:
    if name in {"GrpcTransport", "WorkerRpcServer"}:
        from .grpc_transport import GrpcTransport, WorkerRpcServer

        return {
            "GrpcTransport": GrpcTransport,
            "WorkerRpcServer": WorkerRpcServer,
        }[name]
    if name == "StageRingConnectionPool":
        from .stage_ring_connection import StageRingConnectionPool

        return StageRingConnectionPool
    if name == "StageRingServer":
        from .stage_ring_server import StageRingServer

        return StageRingServer
    raise AttributeError(name)
