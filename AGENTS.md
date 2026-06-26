# AGENTS.md

This repository is a handoff package for running BAGEL BR-KV lifetime experiments on an NVIDIA GPU server.

## Goal

Validate a simple benefit-risk conditioning-KV retirement method for BAGEL-style unified multimodal generation/editing.

The method is intentionally static at inference time:

1. Run offline calibration over role/layer/cutoff cells.
2. Label outputs as `pass`, `minor`, or `fail`.
3. Build BR-KV schedule tables from empirical risk and static benefit.
4. Validate each table as a joint policy.

Do not add online attention classifiers or extra forward passes unless explicitly requested.

## Expected External Dependencies

You need a working BAGEL checkout and the BAGEL-7B-MoT checkpoint.

Recommended layout:

```text
/workspace/BAGEL
/workspace/models/BAGEL-7B-MoT
/workspace/bagel-brkv-lifetime
```

Install the helper scripts into BAGEL first:

```bash
scripts/install_into_bagel.sh /workspace/BAGEL
```

On NVIDIA, keep BAGEL computation semantics unchanged. Only adapt device/dtype defaults and paths.

## Main Commands

Build a schedule from a labeled calibration CSV:

```bash
python -m brkv.build_brkv_schedule \
  --calibration-csv outputs/full/calibration/summary_labeled.csv \
  --output-dir outputs/full/schedules \
  --profiles conservative:0,balanced:1,aggressive:2 \
  --fail-rate-budget 0.10 \
  --suffix-safe
```

Run BAGEL calibration via wrapper:

```bash
python -m brkv.launch_bagel_experiment \
  --config configs/nvidia_full.yaml \
  --bagel-root /workspace/BAGEL \
  --stage calibration
```

Run joint policy validation:

```bash
python -m brkv.launch_bagel_experiment \
  --config configs/nvidia_full.yaml \
  --bagel-root /workspace/BAGEL \
  --stage policy \
  --scheduler-json outputs/full/schedules/balanced_scheduler.json
```

## Experiment Contract

Calibration CSV rows should contain at least:

```text
role, layer_band, cutoff, human_label, zeroed_bytes_upper_bound, retired_step_fraction
```

Accepted labels:

```text
pass
minor
acceptable
fail
unlabeled
```

`unlabeled` rows are ignored by the schedule builder unless `--allow-unlabeled` is set.

## What to Preserve

- Keep BAGEL weights frozen.
- Keep the original edit prompts and source images recorded in outputs.
- Preserve baseline images.
- Keep `T+1` as the no-retirement action.
- Do not claim speedup from zero-masking probes. Speed claims require packed-KV skipping or real attention-kernel work removal.

## Full-Run Checklist

1. Verify BAGEL baseline editing output is correct.
2. Run `configs/nvidia_full.yaml` calibration grid.
3. Label output contact sheets.
4. Build BR-KV profiles.
5. Validate profiles as joint policies.
6. Add matched-benefit baselines if time allows:
   - random cells with same retired benefit
   - static `L24-L27`
   - role-agnostic table
7. Report quality pass rate vs saved KV-attention work.
