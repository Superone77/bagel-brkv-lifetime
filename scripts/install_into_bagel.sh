#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/BAGEL" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAGEL_ROOT="$1"

if [[ ! -d "$BAGEL_ROOT/modeling" || ! -d "$BAGEL_ROOT/data" ]]; then
  echo "Not a BAGEL root: $BAGEL_ROOT" >&2
  exit 2
fi

if [[ -f "$ROOT/bagel_scripts.tar.gz.b64" ]]; then
  tmpdir="$(mktemp -d)"
  base64 -d -i "$ROOT/bagel_scripts.tar.gz.b64" -o "$tmpdir/bagel_scripts.tar.gz" 2>/dev/null \
    || base64 -d "$ROOT/bagel_scripts.tar.gz.b64" > "$tmpdir/bagel_scripts.tar.gz"
  tar -xzf "$tmpdir/bagel_scripts.tar.gz" -C "$tmpdir"
  cp "$tmpdir"/bagel_scripts/*.py "$BAGEL_ROOT"/
  rm -rf "$tmpdir"
else
  cp "$ROOT"/bagel_scripts/*.py "$BAGEL_ROOT"/
fi

PATCH="$ROOT/patches/uni4uni_kv_bagel_hooks.patch"
if [[ -f "$PATCH" ]]; then
  if git -C "$BAGEL_ROOT" apply --check "$PATCH"; then
    git -C "$BAGEL_ROOT" apply "$PATCH"
    echo "Applied Uni4Uni-KV BAGEL hook patch."
  elif git -C "$BAGEL_ROOT" apply --reverse --check "$PATCH"; then
    echo "Uni4Uni-KV BAGEL hook patch already applied."
  else
    echo "Cannot apply Uni4Uni-KV BAGEL hook patch cleanly in $BAGEL_ROOT" >&2
    echo "Please inspect $PATCH and the target BAGEL checkout." >&2
    exit 1
  fi
fi

echo "Installed Uni4Uni-KV BAGEL helper scripts into $BAGEL_ROOT"
