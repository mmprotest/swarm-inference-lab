from __future__ import annotations

import pytest
from build_windows_installer import _verify_pinned_publisher_identity
from release_common import ReleaseError

THUMBPRINT = "E0AB19C8D38CBF9C44709925122A7A02F8C70CB7"
METADATA = {
    "publisher_subject": "CN=Pyrsys B.V., O=Pyrsys B.V., S=Noord-Holland, C=NL",
    "publisher_thumbprint": THUMBPRINT,
}


@pytest.mark.parametrize("status", ["Valid", "UnknownError", "NotTrusted"])
def test_pinned_publisher_accepts_exact_certificate_after_hash_verification(status: str) -> None:
    _verify_pinned_publisher_identity(
        {
            "status": status,
            "subject": METADATA["publisher_subject"],
            "thumbprint": THUMBPRINT.lower(),
        },
        METADATA,
        label="fixture",
    )


@pytest.mark.parametrize("status", ["NotSigned", "HashMismatch", "NotSupportedFileFormat", ""])
def test_pinned_publisher_rejects_missing_or_damaged_signature(status: str) -> None:
    with pytest.raises(ReleaseError, match="publisher verification failed"):
        _verify_pinned_publisher_identity(
            {
                "status": status,
                "subject": METADATA["publisher_subject"],
                "thumbprint": THUMBPRINT,
            },
            METADATA,
            label="fixture",
        )


def test_pinned_publisher_rejects_a_different_signer_certificate() -> None:
    with pytest.raises(ReleaseError, match="publisher verification failed"):
        _verify_pinned_publisher_identity(
            {
                "status": "Valid",
                "subject": METADATA["publisher_subject"],
                "thumbprint": "0" * 40,
            },
            METADATA,
            label="fixture",
        )
