#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo}/.uv-cache}"

backend="auto"
install_only=0
with_dev=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="${2:?--backend requires a value}"
      shift 2
      ;;
    --install-only)
      install_only=1
      shift
      ;;
    --no-dev)
      with_dev=0
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "${backend}" in
  auto|synthetic|cpu|cuda|mps) ;;
  *)
    echo "backend must be one of: auto, synthetic, cpu, cuda, mps" >&2
    exit 2
    ;;
esac

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it with your operating-system package manager." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv; install it with your operating-system package manager." >&2
    exit 2
  fi
  echo "Installing uv for the current user..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ "${install_only}" -eq 1 ]]; then
  exit 0
fi

if [[ "${backend}" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    backend="mps"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    backend="cuda"
  else
    backend="cpu"
  fi
fi

sync_args=(sync)
if [[ "${with_dev}" -eq 1 ]]; then
  sync_args+=(--extra dev)
fi
if [[ "${backend}" != "synthetic" ]]; then
  sync_args+=(--extra "${backend}")
fi

echo "Synchronising native $(uname -s) environment for backend '${backend}'..."
uv "${sync_args[@]}"
uv run --no-sync swarm doctor --backend "${backend}"
