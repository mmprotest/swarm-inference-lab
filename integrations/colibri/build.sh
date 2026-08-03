#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT=b085b48888a88d9a1c00b151a9979774b72cdbfd
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
COLIBRI="$ROOT/third_party/colibri"
OUTPUT="$ROOT/build/colibri"
APPLY=0

while (($#)); do
  case "$1" in
    --colibri-path) COLIBRI=$(cd "$2" && pwd); shift 2 ;;
    --output-directory) OUTPUT=$2; shift 2 ;;
    --apply-bridge-patches) APPLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ACTUAL=$(git -C "$COLIBRI" rev-parse HEAD)
[[ "$ACTUAL" == "$EXPECTED_COMMIT" ]] || {
  echo "Colibri revision mismatch: expected $EXPECTED_COMMIT, found $ACTUAL" >&2
  exit 1
}

rm -rf "$OUTPUT/source"
mkdir -p "$OUTPUT/source" "$OUTPUT/bin"
git -C "$COLIBRI" archive --format=tar "$EXPECTED_COMMIT" | tar -xf - -C "$OUTPUT/source"
PATCHES=()
if ((APPLY)); then
  while IFS= read -r patch; do
    [[ -z "$patch" || "$patch" == \#* ]] && continue
    PATCHES+=("$patch")
    git -C "$OUTPUT/source" apply --check "$ROOT/integrations/colibri/patches/$patch"
    git -C "$OUTPUT/source" apply "$ROOT/integrations/colibri/patches/$patch"
  done < "$ROOT/integrations/colibri/patches/series"
  for adapter in swarm_expert_wire.h swarm_expert_wire.c; do
    test -f "$ROOT/integrations/colibri/adapter/$adapter" || {
      echo "missing canonical C wire adapter: $adapter" >&2
      exit 1
    }
    cp "$ROOT/integrations/colibri/adapter/$adapter" "$OUTPUT/source/c/$adapter"
  done
fi

BUILD_TARGETS=(colibri olmoe olmoe_expert_worker inkling kimi_k3)
NATIVE_TESTS=()
if ((APPLY)); then
  NATIVE_TESTS=(
    tests/test_olmoe_expert_runtime
    tests/test_olmoe_external_dispatch
    tests/test_olmoe_memory_residency
    tests/test_olmoe_expert_shm
  )
  BUILD_TARGETS+=("${NATIVE_TESTS[@]}")
fi
make -C "$OUTPUT/source/c" -j4 "${BUILD_TARGETS[@]}" ARCH=native
gcc -D_FILE_OFFSET_BITS=64 -O3 -march=native -fopenmp -fPIC -shared \
  -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation \
  -Wno-unused-function -I "$OUTPUT/source/c" \
  "$ROOT/integrations/colibri/adapter/kimi_mxfp4_runtime.c" \
  -o "$OUTPUT/source/c/libcoli_kimi_mxfp4.so" -lm -fopenmp
for binary in colibri olmoe olmoe_expert_worker inkling kimi_k3; do
  cp "$OUTPUT/source/c/$binary" "$OUTPUT/bin/$binary"
done
cp "$OUTPUT/source/c/libcoli_kimi_mxfp4.so" "$OUTPUT/bin/libcoli_kimi_mxfp4.so"
cp "$OUTPUT/source/LICENSE" "$OUTPUT/LICENSE.colibri"
for native_test in "${NATIVE_TESTS[@]}"; do
  "$OUTPUT/source/c/$native_test"
done
PATCH_MANIFEST_ARGS=()
if ((APPLY)); then
  PATCH_MANIFEST_ARGS=(
    --patch-directory "$ROOT/integrations/colibri/patches"
    --wire-adapter-directory "$ROOT/integrations/colibri/adapter"
  )
fi
python3 "$ROOT/integrations/colibri/build_manifest.py" \
  --source "$OUTPUT/source" --bin "$OUTPUT/bin" --output "$OUTPUT/colibri_build.json" \
  --commit "$ACTUAL" --patches "${PATCHES[@]}" \
  "${PATCH_MANIFEST_ARGS[@]}" \
  --patch-manifest-output "$OUTPUT/colibri_patch_manifest.json"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q "$ROOT/tests/unit/test_colibri_integration.py"
