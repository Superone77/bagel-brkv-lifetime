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
echo "Installed BR-KV BAGEL helper scripts into $BAGEL_ROOT"
