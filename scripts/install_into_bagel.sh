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

cp "$ROOT"/bagel_scripts/*.py "$BAGEL_ROOT"/
echo "Installed BR-KV BAGEL helper scripts into $BAGEL_ROOT"
