#!/usr/bin/env bash
# tests/unit/scripts/test_update_golden.sh
#
# Drives scripts/update_golden.sh with a fake capture executable and checks the
# resulting goldens directory with the real coast_compare. The property under
# test: every retained golden keeps its OWN binary sidecar, so a later capture
# never changes what an earlier golden matches (they used to share one
# buffer_0.bin, which each capture overwrote). Also covers rotation, cleanup of
# unreferenced sidecars (including a legacy shared buffer_0.bin), stamp
# validation, and refusal of a blob whose reference cannot be resolved.
#
# Usage: test_update_golden.sh <update_golden.sh> <coast_compare>

set -euo pipefail

UPD=$1
CMP=$2
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "ok: $*"; }

GOLDENS="$W/goldens"
case "$(uname -s)" in
    Linux)  TARGET="$GOLDENS" ;;
    Darwin) TARGET="$GOLDENS/macos" ;;
    *)      TARGET="$GOLDENS/$(uname -s | tr 'A-Z' 'a-z')" ;;
esac
mkdir -p "$TARGET"

# One f64 per sidecar, little-endian; printf escapes for the values used below.
f64() {
    case "$1" in
        1) echo '\x00\x00\x00\x00\x00\x00\xf0\x3f' ;;
        2) echo '\x00\x00\x00\x00\x00\x00\x00\x40' ;;
        3) echo '\x00\x00\x00\x00\x00\x00\x08\x40' ;;
        4) echo '\x00\x00\x00\x00\x00\x00\x10\x40' ;;
        5) echo '\x00\x00\x00\x00\x00\x00\x14\x40' ;;
        6) echo '\x00\x00\x00\x00\x00\x00\x18\x40' ;;
        7) echo '\x00\x00\x00\x00\x00\x00\x1c\x40' ;;
        9) echo '\x00\x00\x00\x00\x00\x00\x22\x40' ;;
        *) fail "f64: no encoding for $1" ;;
    esac
}

# write_blob <dir> <value> [<ref>]: a blob whose only array lives in a sidecar.
write_blob() {
    local dir=$1 v=$2 ref=${3:-buffer_0.bin}
    mkdir -p "$dir"
    rm -f "$dir"/*.bin
    printf "$(f64 "$v")" > "$dir/buffer_0.bin"
    printf '{"buffer_0": {"values": {"binary": "%s", "dtype": "f64", "shape": [1]}}}\n' \
        "$ref" > "$dir/blob.json"
}

# The fake test executable update_golden.sh runs (capture-only mode).
cat > "$W/fake_test" <<'EOF'
#!/usr/bin/env bash
set -eu
[ "${COAST_TEST_CAPTURE_ONLY:-}" = 1 ] || { echo "fake_test: expected capture-only" >&2; exit 3; }
mkdir -p "$FAKE_OUT"
rm -f "$FAKE_OUT"/*.bin
printf "$FAKE_BYTES" > "$FAKE_OUT/buffer_0.bin"
printf '{"buffer_0": {"values": {"binary": "%s", "dtype": "f64", "shape": [1]}}}\n' \
    "$FAKE_REF" > "$FAKE_OUT/blob.json"
EOF
chmod +x "$W/fake_test"

capture() {   # capture <stamp> <value> [<ref>]
    COAST_GOLDEN_STAMP=$1 FAKE_BYTES=$(f64 "$2") FAKE_REF=${3:-buffer_0.bin} \
    FAKE_OUT="$W/out" bash "$UPD" "$W/fake_test" "$W/out/blob.json" "$GOLDENS" > /dev/null
}

matches() {   # matches <value>: does some retained golden match this value?
    write_blob "$W/cand" "$1"
    "$CMP" "$W/cand/blob.json" "$TARGET/" > /dev/null 2>&1
}

count() { find "$TARGET" -maxdepth 1 -name "$1" | wc -l | tr -d ' '; }

# --- a legacy golden in the old layout: bare reference to a shared sidecar ---
write_blob "$W/legacy" 9
cp "$W/legacy/blob.json" "$TARGET/20250101_legacy.json"
cp "$W/legacy/buffer_0.bin" "$TARGET/buffer_0.bin"

# --- two captures: each golden keeps its own fingerprint ---
capture 20260101_a 1
capture 20260102_b 2
[ -f "$TARGET/20260101_a.buffer_0.bin" ] || fail "golden a has no sidecar of its own"
[ -f "$TARGET/20260102_b.buffer_0.bin" ] || fail "golden b has no sidecar of its own"
grep -q '"binary": "20260101_a.buffer_0.bin"' "$TARGET/20260101_a.json" \
    || fail "golden a does not reference its own sidecar"
[ "$(stat -c %a "$TARGET/20260101_a.json" 2>/dev/null || stat -f %Lp "$TARGET/20260101_a.json")" = 644 ] \
    || fail "golden JSON is not mode 644"
matches 1 || fail "golden a (value 1) no longer matches after capturing b"
matches 2 || fail "golden b (value 2) does not match"
matches 9 || fail "legacy golden (value 9) stopped matching while still retained"
! matches 5 || fail "a value no golden holds matched"
[ -f "$TARGET/buffer_0.bin" ] || fail "legacy sidecar dropped while its golden is retained"
pass "each golden keeps its own sidecar; legacy layout still resolves"

# --- rotation: 7 goldens -> newest 5; unreferenced sidecars are dropped ---
capture 20260103_c 3
capture 20260104_d 4
capture 20260105_e 5
capture 20260106_f 6
[ "$(count '*.json')" = 5 ] || fail "expected 5 goldens after rotation, found $(count '*.json')"
[ ! -e "$TARGET/20250101_legacy.json" ] || fail "oldest (legacy) golden not rotated out"
[ ! -e "$TARGET/20260101_a.json" ] || fail "second-oldest golden not rotated out"
[ ! -e "$TARGET/buffer_0.bin" ] || fail "legacy shared sidecar kept after its last golden left"
[ ! -e "$TARGET/20260101_a.buffer_0.bin" ] || fail "rotated-out golden's sidecar kept"
[ "$(count '*.bin')" = 5 ] || fail "expected 5 sidecars after rotation, found $(count '*.bin')"
! matches 1 || fail "rotated-out golden still matches"
matches 2 || fail "retained golden b no longer matches after rotation"
matches 6 || fail "newest golden does not match"
pass "rotation keeps 5 goldens and exactly their sidecars"

# --- re-capturing the same stamp replaces that golden in place ---
capture 20260106_f 7
[ "$(count '*.json')" = 5 ] && [ "$(count '*.bin')" = 5 ] || fail "re-capture changed the golden count"
matches 7 || fail "re-captured golden does not match its new value"
! matches 6 || fail "re-captured golden still matches its old value"
pass "re-capture with the same stamp is in place"

# --- refusals: nothing may be written, not even a sidecar ---
snapshot() { (cd "$TARGET" && ls -A && cksum -- *); }   # names (dotfiles too) + contents
before=$(snapshot)
if capture 'bad/stamp' 1 2> /dev/null; then fail "an invalid stamp was accepted"; fi
if capture 20260107_g 1 elsewhere.bin 2> /dev/null; then
    fail "a blob referencing a missing sidecar was stored"
fi
# Same stamp as an existing golden (same day + same commit): the refused data
# must not reach that golden's sidecar.
if capture 20260106_f 1 elsewhere.bin 2> /dev/null; then
    fail "a same-stamp blob referencing a missing sidecar was stored"
fi
[ "$before" = "$(snapshot)" ] || fail "refused captures changed the goldens directory"
matches 7 || fail "a refused same-stamp capture changed the existing golden's fingerprint"
! matches 1 || fail "a refused capture's data became matchable"
pass "invalid stamp and unresolvable references are refused without side effects"

echo "ALL PASS"
