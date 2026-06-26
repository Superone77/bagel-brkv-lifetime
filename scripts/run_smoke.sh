#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m brkv.build_brkv_schedule \
  --calibration-csv examples/labeled_calibration_smoke.csv \
  --output-dir outputs/schedule_smoke \
  --profiles conservative:0,balanced:1,aggressive:2 \
  --fail-rate-budget 0.10 \
  --suffix-safe

python -m brkv.launch_bagel_experiment \
  --config configs/audrey_smoke.yaml \
  --bagel-root ../BAGEL \
  --stage print
