#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

DEST="${1:-$SWARM_HOME/collected/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$DEST"
for SOURCE in "$SWARM_HOME/canary" "$SWARM_HOME/certificates" "$SWARM_HOME/coordinator"; do
  if test -d "$SOURCE"; then cp -a "$SOURCE" "$DEST/"; fi
done
find "$SWARM_HOME/workers" -path '*/receipts/*.json' -type f -exec cp --parents '{}' "$DEST/" \; 2>/dev/null || true
find "$DEST" -type f -exec sha256sum '{}' \; | sort > "$DEST/SHA256SUMS"
echo "$DEST"
