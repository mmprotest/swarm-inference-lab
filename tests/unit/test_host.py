from __future__ import annotations

import pytest

from swarm_inference.exceptions import ConfigurationError
from swarm_inference.host import (
    endpoint_is_local,
    format_endpoint,
    is_loopback_host,
    qualifies_as_remote_physical_worker,
    resolve_advertised_endpoint,
    split_endpoint,
)


def test_endpoint_parsing_is_platform_neutral() -> None:
    assert split_endpoint("worker.local:50052") == ("worker.local", 50052)
    assert split_endpoint("[::1]:50052") == ("::1", 50052)
    assert format_endpoint("::1", 50052) == "[::1]:50052"
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert endpoint_is_local("127.0.0.1:50052")


def test_wildcard_listen_derives_loopback_advertisement() -> None:
    assert (
        resolve_advertised_endpoint(
            listen_endpoint="0.0.0.0:50052",
            coordinator_endpoint="127.0.0.1:50051",
            explicit_endpoint=None,
        )
        == "127.0.0.1:50052"
    )


def test_wildcard_advertisement_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="not a wildcard"):
        resolve_advertised_endpoint(
            listen_endpoint="0.0.0.0:50052",
            coordinator_endpoint="coordinator.local:50051",
            explicit_endpoint="0.0.0.0:50052",
        )


def test_physical_worker_requires_remote_hostname_and_address() -> None:
    assert qualifies_as_remote_physical_worker(
        worker_hostname="remote-node",
        endpoint="192.0.2.10:50052",
        coordinator_hostname="coordinator-node",
    )
    assert not qualifies_as_remote_physical_worker(
        worker_hostname="coordinator-node",
        endpoint="192.0.2.10:50052",
        coordinator_hostname="coordinator-node",
    )
    assert not qualifies_as_remote_physical_worker(
        worker_hostname="remote-node",
        endpoint="127.0.0.1:50052",
        coordinator_hostname="coordinator-node",
    )


@pytest.mark.parametrize("endpoint", ["missing-port", "::1:50052", "host:70000"])
def test_invalid_endpoint_fails_precisely(endpoint: str) -> None:
    with pytest.raises(ConfigurationError):
        split_endpoint(endpoint)
