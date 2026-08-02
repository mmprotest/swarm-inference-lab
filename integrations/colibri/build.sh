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
fi

make -C "$OUTPUT/source/c" -j4 colibri olmoe inkling kimi_k3 ARCH=native
for binary in colibri olmoe inkling kimi_k3; do
  cp "$OUTPUT/source/c/$binary" "$OUTPUT/bin/$binary"
done
cp "$OUTPUT/source/LICENSE" "$OUTPUT/LICENSE.colibri"
python3 "$ROOT/integrations/colibri/build_manifest.py" \
  --source "$OUTPUT/source" --bin "$OUTPUT/bin" --output "$OUTPUT/colibri_build.json" \
  --commit "$ACTUAL" --patches "${PATCHES[@]}"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q "$ROOT/tests/unit/test_colibri_integration.py"

