#!/bin/sh
set -eu

source_wheel=""
install_service=0
json_output=0
uv_path=""
python_version="3.11"
installer=""
doctor_file=""

cleanup() {
    [ -z "$installer" ] || rm -f "$installer"
    [ -z "$doctor_file" ] || rm -f "$doctor_file"
}
trap cleanup EXIT HUP INT TERM

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source-wheel) source_wheel=$2; shift 2 ;;
        --install-service) install_service=1; shift ;;
        --json) json_output=1; shift ;;
        --uv-path) uv_path=$2; shift 2 ;;
        --python-version) python_version=$2; shift 2 ;;
        *) printf '%s\n' "unknown installer option: $1" >&2; exit 2 ;;
    esac
done

case "$python_version" in 3.11|3.12|3.13) ;; *) printf '%s\n' "unsupported Python: $python_version" >&2; exit 2 ;; esac
export UV_HTTP_TIMEOUT=120
export UV_CONCURRENT_DOWNLOADS=8

run_bounded() {
    limit=$1
    shift
    # The explicit stdin duplication is required: POSIX asynchronous commands
    # may otherwise inherit /dev/null, which breaks the bounded JSON parsers
    # used in pipelines and with the final here-document.
    "$@" <&0 &
    command_pid=$!
    elapsed=0
    timed_out=0
    while kill -0 "$command_pid" 2>/dev/null; do
        if [ "$elapsed" -ge "$limit" ]; then
            timed_out=1
            kill -TERM "$command_pid" 2>/dev/null || true
            grace=0
            while kill -0 "$command_pid" 2>/dev/null && [ "$grace" -lt 5 ]; do
                sleep 1
                grace=$((grace + 1))
            done
            kill -KILL "$command_pid" 2>/dev/null || true
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    set +e
    wait "$command_pid"
    status=$?
    set -e
    if [ "$timed_out" -eq 1 ]; then
        return 124
    fi
    return "$status"
}

fail() {
    detail=$1
    if [ "$json_output" -eq 1 ]; then
        escaped=$(printf '%s' "$detail" | sed 's/\\/\\\\/g; s/"/\\"/g')
        printf '{"schema_version":1,"status":"FAIL","stage":"installation","category":"execution","detail":"%s","retry_safe":true}\n' "$escaped"
    else
        printf 'installation failed: %s\n' "$detail" >&2
    fi
    exit 1
}

if [ -z "$uv_path" ]; then
    uv_path=$(command -v uv || true)
fi
if [ -z "$uv_path" ]; then
    installer=$(mktemp "${TMPDIR:-/tmp}/swarm-uv-install.XXXXXX")
    if command -v curl >/dev/null 2>&1; then
        curl --fail --silent --show-error --max-time 120 https://astral.sh/uv/install.sh -o "$installer" || fail "uv download failed"
    elif command -v wget >/dev/null 2>&1; then
        wget --timeout=120 -q https://astral.sh/uv/install.sh -O "$installer" || fail "uv download failed"
    else
        fail "curl or wget is required to install uv"
    fi
    run_bounded 300 sh "$installer" || fail "uv installation failed"
    uv_path="$HOME/.local/bin/uv"
fi
[ -x "$uv_path" ] || fail "uv executable is unavailable"
run_bounded 900 "$uv_path" python install "$python_version" || fail "managed Python installation failed"

if [ -n "$source_wheel" ]; then
    [ -f "$source_wheel" ] || fail "--source-wheel does not exist"
    case "$source_wheel" in *.whl) ;; *) fail "--source-wheel must name a built wheel" ;; esac
    package=$(cd "$(dirname "$source_wheel")" && pwd)/$(basename "$source_wheel")
    source_kind="source-wheel"
else
    package="swarm-inference-lab"
    source_kind="package-index"
fi

system=$(uname -s)
architecture=$(uname -m)
case "$system:$architecture" in
    Linux:x86_64|Linux:amd64|Linux:aarch64|Linux:arm64|Darwin:arm64|Darwin:aarch64) ;;
    *) fail "unsupported operating system or architecture: $system $architecture" ;;
esac
candidate="cpu"
torch_backend="cpu"
if [ "$system" = "Darwin" ] && { [ "$architecture" = "arm64" ] || [ "$architecture" = "aarch64" ]; }; then
    candidate="mps"
    torch_backend="auto"
elif command -v nvidia-smi >/dev/null 2>&1 && run_bounded 20 nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    candidate="cuda"
    torch_backend="cu130"
fi

install_tool() {
    extra=$1
    backend=$2
    run_bounded 1800 "$uv_path" tool install --force --python "$python_version" --torch-backend "$backend" "$package[$extra]"
}

install_tool "$candidate" "$torch_backend" || fail "wheel tool installation failed"
bin_directory=$(run_bounded 30 "$uv_path" tool dir --bin) || fail "uv tool bin lookup failed"
swarm_executable="$bin_directory/swarm"
[ -x "$swarm_executable" ] || fail "installed swarm executable was not found"
doctor=$(run_bounded 180 "$swarm_executable" node doctor --json) || fail "swarm node doctor failed"
doctor_file=$(mktemp "${TMPDIR:-/tmp}/swarm-doctor.XXXXXX")
chmod 600 "$doctor_file"
printf '%s\n' "$doctor" >"$doctor_file"
selected=$(run_bounded 60 "$uv_path" run --python "$python_version" python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["backend_selection"]["selected_backend"])' "$doctor_file") || fail "doctor result parsing failed"
selected_extra="cpu"
[ "$selected" = "torch-cuda" ] && selected_extra="cuda"
[ "$selected" = "torch-mps" ] && selected_extra="mps"
if [ "$selected_extra" != "$candidate" ]; then
    fallback_backend="cpu"
    [ "$selected_extra" = "mps" ] && fallback_backend="auto"
    install_tool "$selected_extra" "$fallback_backend" || fail "operational-backend reinstall failed"
    doctor=$(run_bounded 180 "$swarm_executable" node doctor --json) || fail "post-fallback node doctor failed"
    printf '%s\n' "$doctor" >"$doctor_file"
    selected=$(run_bounded 60 "$uv_path" run --python "$python_version" python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["backend_selection"]["selected_backend"])' "$doctor_file") || fail "fallback doctor result parsing failed"
fi

service="deferred-until-cluster-create-or-join"
if [ "$install_service" -eq 1 ]; then
    service_preference="requested-deferred"
else
    service_preference="default-deferred"
fi

if [ "$json_output" -eq 1 ]; then
    run_bounded 60 "$uv_path" run --python "$python_version" python -c 'import json,sys; print(json.dumps({"schema_version":1,"status":"PASS","operating_system":sys.argv[1],"architecture":sys.argv[2],"python_version":sys.argv[3],"source":sys.argv[4],"package_extra":sys.argv[5],"selected_backend":sys.argv[6],"swarm_executable":sys.argv[7],"service":sys.argv[8],"install_service_preference":sys.argv[9],"doctor":json.load(open(sys.argv[10], encoding="utf-8"))},sort_keys=True,separators=(",",":")))' "$system" "$architecture" "$python_version" "$source_kind" "$selected_extra" "$selected" "$swarm_executable" "$service" "$service_preference" "$doctor_file"
else
    printf 'status=PASS\nswarm_executable=%s\nselected_backend=%s\nservice=%s\n' "$swarm_executable" "$selected" "$service"
fi
