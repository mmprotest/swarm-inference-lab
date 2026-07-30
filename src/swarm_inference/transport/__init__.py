"""Transport abstraction and gRPC AsyncIO implementation."""

from typing import TYPE_CHECKING, Any

from .base import ActivationTransport

if TYPE_CHECKING:
    from .grpc_transport import GrpcTransport, WorkerRpcServer

__all__ = ["ActivationTransport", "GrpcTransport", "WorkerRpcServer"]


def __getattr__(name: str) -> Any:
    if name in {"GrpcTransport", "WorkerRpcServer"}:
        from .grpc_transport import GrpcTransport, WorkerRpcServer

        return {
            "GrpcTransport": GrpcTransport,
            "WorkerRpcServer": WorkerRpcServer,
        }[name]
    raise AttributeError(name)
