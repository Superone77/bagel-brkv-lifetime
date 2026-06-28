# BAGEL Uni4Uni-KV Experiments

This repository packages the BAGEL helper scripts and patch needed to run **Uni4Uni-KV** experiments on a GPU server.

Uni4Uni-KV is a unified KV-cache lifetime / TopK budgeting probe for BAGEL-style unified multimodal models. It currently masks non-selected candidate KV tokens before attention and reports theoretical skipped KV work. It is a quality and feasibility probe, not a speedup implementation.

## What Is Included

```text
bagel_scripts/
  demo_mps.py                         # BAGEL model loading helper with cuda/mps/cpu support.
  run_pyramid_topk_kv_demo.py          # Three-task Uni4Uni-KV demo.
  run_uni4uni_kv_sweep.py              # Parameter sweep wrapper.
  analyze_*.py                         # Older local ablation helpers.
patches/
  uni4uni_kv_bagel_hooks.patch         # BAGEL attention/context hooks required by the demo.
scripts/
  install_into_bagel.sh                # Copy scripts into BAGEL and apply the hook patch.
brkv/
  build_brkv_schedule.py               # Older BR-KV schedule builder, kept for reference.
configs/, examples/
  Older calibration examples.
AGENTS.md                             # GPU-server handoff instructions.
```

## Install Into BAGEL

Recommended server layout:

```text
/workspace/BAGEL
/workspace/models/BAGEL-7B-MoT
/workspace/bagel-brkv-lifetime
```

Install:

```bash
cd /workspace/bagel-brkv-lifetime
scripts/install_into_bagel.sh /workspace/BAGEL
```

The installer copies the helper scripts into the BAGEL root and applies `patches/uni4uni_kv_bagel_hooks.patch`. If the patch has already been applied, it reports that and continues.

## Uni4Uni-KV Demo

Run one full-KV baseline and one Uni4Uni-KV variant on three tasks:

1. Understanding: read visible text in `test_images/meme.jpg`.
2. Text-to-image: long prompt for a rocket-powered raccoon astronaut.
3. Image editing: edit `test_images/women.jpg` so the outfit becomes vivid blue and the background becomes a snowy mountain lake.

```bash
cd /workspace/BAGEL
python run_pyramid_topk_kv_demo.py \
  --device cuda \
  --dtype bfloat16 \
  --image-size 512 \
  --timesteps 50 \
  --max-new-tokens 256 \
  --beta 20 \
  --alpha-understanding 8 \
  --understanding-keep-ratio 0.5 \
  --gen-keep-start 0.8 \
  --gen-keep-end 0.1 \
  --output-dir outputs/uni4uni_kv_demo_b20_a8_s80_e10
```

Main outputs:

```text
summary.json
run_report.md
topk_stats.csv
baseline_generation.png
topk_generation.png
baseline_editing.png
topk_editing.png
```

## Parameter Sweep

Standard 50-step sweep:

```bash
cd /workspace/BAGEL
python run_uni4uni_kv_sweep.py \
  --device cuda \
  --dtype bfloat16 \
  --image-size 512 \
  --timesteps 50 \
  --max-new-tokens 256 \
  --betas 8,14,16,20 \
  --alphas 8,16 \
  --schedules '0.8->0.1,0.8->0.2,0.9->0.2,0.9->0.3' \
  --understanding-keep-ratio 0.5 \
  --output-root outputs/uni4uni_kv_param_sweep_512_t50
```

Smoke test:

```bash
cd /workspace/BAGEL
python run_uni4uni_kv_sweep.py \
  --device cuda \
  --dtype bfloat16 \
  --image-size 320 \
  --timesteps 6 \
  --max-new-tokens 96 \
  --betas 20 \
  --alphas 8 \
  --schedules '0.8->0.1' \
  --limit 1 \
  --output-root outputs/uni4uni_kv_smoke
```

## Policy Details

Uni4Uni-KV uses attention scores to select TopK candidate KV tokens under a layer-wise pyramid budget.

For understanding:

```text
mean keep ratio = 0.5
beta controls shallow-to-deep layer budget decay
alpha_understanding protects recent text tokens
```

For generation and editing:

```text
mean keep ratio decays across denoising steps, e.g. 0.8 -> 0.1
beta controls shallow-to-deep layer budget decay
current VAE latent/self KV is always protected
only conditioning/past KV is budgeted
```

PyramidKV reference values:

```text
beta = 20
alpha = 8
```

Local sweep values:

```text
beta in {8, 14, 16, 20}
alpha_understanding in {8, 16}
generation schedules in {0.8->0.1, 0.8->0.2, 0.9->0.2, 0.9->0.3}
```

## Caveat

The current implementation masks KV visibility and computes TopK online. It can validate whether quality survives KV budgeting, but it should not be used as evidence of wall-clock speedup. Speedup requires replacing the mask probe with real pre-attention KV compaction or attention-kernel work skipping.

## Older BR-KV Utilities

Older static lifecycle calibration utilities remain in `brkv/`, `configs/`, and `examples/`. They can still build schedule tables from labeled calibration CSVs, but they are no longer the main path for the current Uni4Uni-KV GPU experiments.
