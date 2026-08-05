#!/bin/sh
set -eu

purge=0
yes=0
json=0

run_bounded() {
    limit=$1
    shift
    "$@" &
    command_pid=$!
    (
        sleep "$limit"
        kill -TERM "$command_pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$command_pid" 2>/dev/null || true
    ) &
    watchdog_pid=$!
    set +e
    wait "$command_pid"
    status=$?
    set -e
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    return "$status"
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --purge-state) purge=1; shift ;;
        --yes) yes=1; shift ;;
        --json) json=1; shift ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if command -v swarm >/dev/null 2>&1; then
    run_bounded 180 swarm node uninstall-service --yes --json >/dev/null 2>&1 || true
fi
run_bounded 300 uv tool uninstall swarm-inference-lab >/dev/null
purged=false
if [ "$purge" -eq 1 ]; then
    [ "$yes" -eq 1 ] || { printf '%s\n' "--purge-state requires --yes" >&2; exit 2; }
    case "$(uname -s)" in
        Darwin) state="$HOME/Library/Application Support/SwarmInference" ;;
        *) state="${XDG_STATE_HOME:-$HOME/.local/state}/swarm-inference" ;;
    esac
    [ -n "$state" ] && [ "$state" != "$HOME" ] && [ "$state" != "/" ] || { printf '%s\n' "unsafe state path" >&2; exit 2; }
    run_bounded 300 rm -rf -- "$state"
    purged=true
fi
if [ "$json" -eq 1 ]; then
    printf '{"schema_version":1,"status":"PASS","state_purged":%s}\n' "$purged"
else
    printf 'status=PASS\nstate_purged=%s\n' "$purged"
fi
