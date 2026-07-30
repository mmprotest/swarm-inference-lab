#!/usr/bin/env bash
set -euo pipefail

echo "bootstrap_wsl.sh is retained as a compatibility alias; using the OS-neutral bootstrap."
exec bash "$(dirname "$0")/bootstrap.sh" "$@"
