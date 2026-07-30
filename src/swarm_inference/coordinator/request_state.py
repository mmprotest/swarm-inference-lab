"""Concurrency-safe request state store."""

from __future__ import annotations

import threading

from swarm_inference.config.models import RequestState, RequestStatus, VerificationState
from swarm_inference.exceptions import ConfigurationError


class RequestStore:
    def __init__(self) -> None:
        self._requests: dict[str, RequestState] = {}
        self._lock = threading.RLock()

    def create(self, request: RequestState) -> None:
        with self._lock:
            if request.request_id in self._requests:
                raise ConfigurationError(f"duplicate request ID {request.request_id}")
            self._requests[request.request_id] = request.model_copy(deep=True)

    def get(self, request_id: str) -> RequestState:
        with self._lock:
            try:
                return self._requests[request_id].model_copy(deep=True)
            except KeyError as exc:
                raise ConfigurationError(f"unknown request {request_id}") from exc

    def commit_token(self, request_id: str, token_id: int) -> RequestState:
        with self._lock:
            request = self._requests[request_id]
            if request.status in {
                RequestStatus.COMPLETED,
                RequestStatus.CANCELLED,
                RequestStatus.FAILED,
            }:
                raise ConfigurationError(
                    f"cannot commit token to request in {request.status.value} state"
                )
            request.status = RequestStatus.RUNNING
            request.committed_output_tokens.append(token_id)
            request.current_token_position += 1
            return request.model_copy(deep=True)

    def finish(
        self,
        request_id: str,
        *,
        verified: bool,
        failed: bool = False,
    ) -> RequestState:
        with self._lock:
            request = self._requests[request_id]
            request.status = RequestStatus.FAILED if failed else RequestStatus.COMPLETED
            request.verification_state = (
                VerificationState.REJECTED
                if failed
                else VerificationState.VERIFIED
                if verified
                else VerificationState.UNVERIFIED
            )
            return request.model_copy(deep=True)

    def cancel(self, request_id: str) -> RequestState:
        with self._lock:
            request = self._requests[request_id]
            request.status = RequestStatus.CANCELLED
            return request.model_copy(deep=True)
