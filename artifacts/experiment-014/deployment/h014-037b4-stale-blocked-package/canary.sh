#!/usr/bin/env bash
set -euo pipefail

echo "ABORT: Experiment 014 pre-cluster certification is FAIL." >&2
echo "No fleet may be activated while DEPLOYMENT-BLOCKED.json exists." >&2
echo "Resolve the production Kimi sm_86 and distributed-stage blockers, then regenerate this package." >&2
exit 78
