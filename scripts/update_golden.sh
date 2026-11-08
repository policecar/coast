#!/usr/bin/env bash
# update_golden.sh — capture a golden blob for one integration test.
#
# Usage:
#   update_golden.sh <test_exe> <blob_path> <goldens_dir>
#
# Steps:
#   1. Run <test_exe> with COAST_TEST_CAPTURE_ONLY=1 so it produces its blob
#      but does NOT call the comparator (no golden needed yet).
#   2. Copy the produced blob and any .bin sidecars next to it into
#      <goldens_dir>/ (Linux) or <goldens_dir>/<platform>/ (non-Linux),
#      naming the JSON file <stamp>.json with <stamp> = YYYYMMDD_<short-sha>.
#      Each sidecar is stored under its golden's own name, <stamp>.<name>.bin,
#      and the JSON's "binary" references are rewritten to match — so every
#      retained golden keeps its own fingerprint (a shared buffer_0.bin would
#      be overwritten by every capture, silently re-pointing older goldens).
#   3. Keep the newest 5 *.json files in the target dir; drop older ones, then
#      delete every .bin no remaining golden references.
#
# Environment:
#   COAST_GOLDEN_STAMP  override <stamp> (tests; must match [A-Za-z0-9_-]+)
#
# Platform layout:
#   Linux  → goldens/              (flat, preserves committed Linux goldens)
#   Darwin → goldens/macos/        (subdir, ignored by .gitignore)
#   Other  → goldens/<os>/         (subdir, ignored by .gitignore)
#
# Exits non-zero if any step fails.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <test_exe> <blob_path> <goldens_dir>" >&2
    exit 2
fi

EXE=$1
BLOB=$2
GOLDENS=$3

if [ ! -x "$EXE" ]; then
    echo "update_golden.sh: test executable not executable: $EXE" >&2
    exit 2
fi

mkdir -p "$GOLDENS"

# 1. Capture
echo "Capturing blob via $EXE ..."
COAST_TEST_CAPTURE_ONLY=1 "$EXE"

if [ ! -f "$BLOB" ]; then
    echo "update_golden.sh: expected blob not produced: $BLOB" >&2
    exit 2
fi

# 2. Resolve identifiers and compute target directory.
# Script lives at <repo>/scripts/, so repo root is one level up.
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATE=$(date -u +%Y%m%d)
SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "nogit")
STAMP=${COAST_GOLDEN_STAMP:-"${DATE}_${SHA}"}
if [[ ! "$STAMP" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "update_golden.sh: invalid stamp '$STAMP' (allowed: [A-Za-z0-9_-]+)" >&2
    exit 2
fi
NEW_NAME="${STAMP}.json"

# Platform detection: Linux goldens stay flat in $GOLDENS (preserving the
# existing committed goldens untouched and gitignore-compatible). Non-Linux
# goldens go into a platform subdir so that (a) gitignore can exclude them
# with a single tests/integration/*/goldens/*/ pattern, (b) a macOS regen
# cannot prune the canonical Linux goldens, and (c) sidecar .bin files are
# co-located with their JSON in the same dir so relative references stay valid.
case "$(uname -s)" in
    Linux)  PLATFORM=linux  ;;
    Darwin) PLATFORM=macos  ;;
    *)      PLATFORM=$(uname -s | tr 'A-Z' 'a-z') ;;
esac

if [ "$PLATFORM" = "linux" ]; then
    TARGET="$GOLDENS"
else
    TARGET="${GOLDENS}/${PLATFORM}"
    mkdir -p "$TARGET"
fi

NEW_PATH="${TARGET}/${NEW_NAME}"

# The "binary" references of a golden JSON, one per line. A reference is the
# sidecar's file name; the comparator resolves it relative to the JSON's dir.
binary_refs() {
    grep -oE '"binary"[[:space:]]*:[[:space:]]*"[^"]*"' "$1" \
        | sed -E 's/^.*"([^"]*)"$/\1/' || true
}

# Stage the new golden — each .bin sidecar as <stamp>.<name>.bin, and the JSON
# with its references pointed at them — in a scratch dir inside TARGET, and move
# it into place only once it has been validated. Nothing touches TARGET before
# then, so a refused capture leaves every existing golden as it was (including
# one with the same stamp, whose sidecars a direct copy would overwrite).
# Co-locating json+bin in TARGET keeps the references valid for both flat
# (Linux) and subdir (non-Linux) layouts; the stamp prefix keeps each golden's
# sidecars its own. Staging inside TARGET keeps the final mv a same-filesystem
# rename.
BLOB_DIR=$(dirname "$BLOB")
STAGE=$(mktemp -d "${TARGET}/.stage.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
TMP_JSON="${STAGE}/${NEW_NAME}"
cp "$BLOB" "$TMP_JSON"
shopt -s nullglob
for f in "${BLOB_DIR}"/*.bin; do
    base=$(basename "$f")
    if [[ ! "$base" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "update_golden.sh: unsupported sidecar name '$base'" >&2
        exit 2
    fi
    dest="${STAMP}.${base}"
    cp "$f" "${STAGE}/${dest}"
    esc=${base//./\\.}   # the name's only regex metacharacter is '.'
    sed -E "s/(\"binary\"[[:space:]]*:[[:space:]]*\")${esc}(\")/\\1${dest}\\2/g" \
        "$TMP_JSON" > "$TMP_JSON.sed"
    mv "$TMP_JSON.sed" "$TMP_JSON"
done
shopt -u nullglob

# Fail rather than store a golden whose fingerprint lives elsewhere: every
# reference must be one of this golden's own sidecars, and must exist.
while IFS= read -r ref; do
    [ -z "$ref" ] && continue
    if [[ "$ref" != "${STAMP}."* ]] || [ ! -f "${STAGE}/${ref}" ]; then
        echo "update_golden.sh: reference '${ref}' in the captured blob does not" \
             "resolve to one of its sidecars in ${BLOB_DIR}; nothing written" >&2
        exit 2
    fi
done < <(binary_refs "$TMP_JSON")

# Promote: sidecars first, the JSON last, so a golden never references a
# sidecar that is not yet in place.
shopt -s nullglob
for b in "$STAGE"/*.bin; do
    mv "$b" "${TARGET}/"
    echo "Wrote ${TARGET}/$(basename "$b")"
done
shopt -u nullglob
chmod 644 "$TMP_JSON"   # goldens are ordinary files, whatever the blob's mode
mv "$TMP_JSON" "$NEW_PATH"
echo "Wrote ${NEW_PATH}"

# 3. Rotate: keep newest 5 *.json (β naming = lexicographic = chronological).
# Scope to TARGET so a macOS regen never deletes flat Linux goldens.
mapfile -t JSONS < <(ls -1 "$TARGET"/*.json 2>/dev/null | sort)
COUNT=${#JSONS[@]}
if [ "$COUNT" -gt 5 ]; then
    DROP=$((COUNT - 5))
    echo "Rotating: dropping $DROP oldest golden(s)"
    for ((i = 0; i < DROP; ++i)); do
        echo "  drop ${JSONS[$i]}"
        rm -f "${JSONS[$i]}"
    done
fi

# 4. Drop sidecars no remaining golden references: those of rotated-out
# goldens, and a legacy shared buffer_0.bin once its last golden is gone.
declare -A REFERENCED=()
for j in "$TARGET"/*.json; do
    [ -e "$j" ] || continue
    while IFS= read -r ref; do
        [ -n "$ref" ] && REFERENCED["$ref"]=1
    done < <(binary_refs "$j")
done
for b in "$TARGET"/*.bin; do
    [ -e "$b" ] || continue
    if [ -z "${REFERENCED[$(basename "$b")]:-}" ]; then
        echo "  drop unreferenced sidecar $b"
        rm -f "$b"
    fi
done

echo "Done."
