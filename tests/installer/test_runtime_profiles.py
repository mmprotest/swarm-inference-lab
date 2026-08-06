from __future__ import annotations

from pathlib import Path

import pytest
from generate_runtime_profiles import _normalise_export
from release_common import ReleaseError

HASH = "a" * 64


def _requirement(name: str, version: str) -> str:
    return f"{name}=={version} \\\n+    --hash=sha256:{HASH}"


def test_cpu_profile_uses_pinned_cpu_source_without_development_dependencies() -> None:
    profile = _normalise_export(_requirement("torch", "2.11.0+cpu"), "cpu")
    assert "--index-url https://download.pytorch.org/whl/cpu" in profile
    assert "--extra-index-url https://pypi.org/simple" in profile
    assert "+cu130" not in profile
    assert "pytest==" not in profile
    assert "swarm-inference-lab==" not in profile


def test_cuda_profile_uses_pinned_cuda_source_and_windows_triton() -> None:
    raw = "\n".join(
        [_requirement("torch", "2.11.0+cu130"), _requirement("triton-windows", "3.7.1.post27")]
    )
    profile = _normalise_export(raw, "cuda")
    assert "--index-url https://download.pytorch.org/whl/cu130" in profile
    assert "triton-windows==3.7.1.post27" in profile
    assert "--require-hashes" in profile


@pytest.mark.parametrize(
    "raw",
    [
        f"pytest==9.0.0 \\\n+    --hash=sha256:{HASH}\n{_requirement('torch', '2.11.0+cpu')}",
        f"-e file:///checkout/project\n{_requirement('torch', '2.11.0+cpu')}",
    ],
)
def test_profile_rejects_development_and_checkout_inputs(raw: str) -> None:
    with pytest.raises(ReleaseError):
        _normalise_export(raw, "cpu")


def test_generator_contract_is_locked_and_toolchain_is_hash_pinned(repository_root: Path) -> None:
    source = (repository_root / "scripts/generate_runtime_profiles.py").read_text(encoding="utf-8")
    for switch in ("--locked", "--no-dev", "--no-emit-project", "--no-emit-local"):
        assert switch in source
    toolchain = (repository_root / "installer/windows/toolchain.json").read_text(encoding="utf-8")
    assert '"version": "0.12.0"' in toolchain
    assert '"executable_sha256": "sha256:' in toolchain
    assert '"compiler_sha256": "sha256:' in toolchain
