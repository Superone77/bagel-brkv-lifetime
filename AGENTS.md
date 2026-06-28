# AGENTS.md

This repository is a handoff package for running BAGEL Uni4Uni-KV experiments on an NVIDIA GPU server.

## Current Goal

Run and evaluate **Uni4Uni-KV**, a unified KV-cache lifetime / TopK budgeting method for BAGEL-style unified multimodal models.

The current probe keeps BAGEL weights and core computation unchanged. It adds an inference-time attention mask before attention so only selected candidate KV tokens are visible. This is a correctness and quality probe; it reports theoretical skipped KV work, not guaranteed speedup. Real latency claims require physical KV compaction or kernel-level skipping.

## Method Name

Use the name:

```text
Uni4Uni-KV
```

Historical aliases in older notes/code include `BR-KV` and `Unified Pyramid TopK-KV`. Prefer `Uni4Uni-KV` in new reports.

## Core Policy

For each attention layer, candidate KV tokens are ranked by attention probability:

```text
score(token) = mean over heads and queries of attention_prob(query, token)
```

The layer keeps TopK candidate KV tokens under a pyramid budget. Current active self/latent KV is protected and must not be evicted.

### Understanding

Only the understanding branch is active.

```text
mean_keep_ratio = 0.5
layer budget: shallow layers keep more, deep layers keep less
alpha_understanding protects the most recent text KV tokens
```

### Generation / Editing

The generation branch denoising loop is active, conditioned on understanding-side KV.

```text
mean_keep_ratio decays over denoising steps, e.g. 0.8 -> 0.1
layer budget: shallow layers keep more, deep layers keep less
all current VAE latent/self KV is protected
only conditioning/past KV is budgeted
```

Do not delete or mask the active VAE latent KV. Bad image quality usually means the implementation accidentally budgeted latent/self KV instead of only conditioning KV.

## PyramidKV-Inspired Parameters

Reference values from PyramidKV:

```text
beta = 20
alpha = 8
```

Local interpretation for BAGEL:

```text
beta: layer-pyramid shape controller for all tasks
alpha_understanding: recent-token protection for understanding/text generation only
alpha_gen/edit: replaced by hard protection of active VAE latent/self KV
```

Recommended GPU sweep:

```text
beta in {8, 14, 16, 20}
alpha_understanding in {8, 16}
generation schedules in {0.8->0.1, 0.8->0.2, 0.9->0.2, 0.9->0.3}
```

## Expected Layout

```text
/workspace/BAGEL
/workspace/models/BAGEL-7B-MoT
/workspace/bagel-brkv-lifetime
```

Install helper scripts and BAGEL hooks:

```bash
cd /workspace/bagel-brkv-lifetime
scripts/install_into_bagel.sh /workspace/BAGEL
```

The installer copies `bagel_scripts/*.py` into the BAGEL root and applies:

```text
patches/uni4uni_kv_bagel_hooks.patch
```

If the patch is already applied, the installer should report that and continue.

## Three-Task Demo

Run a full-KV baseline and one Uni4Uni-KV variant over three tasks:

1. Understanding: read text from `test_images/meme.jpg`.
2. Text-to-image: long prompt for a rocket-powered raccoon astronaut.
3. Image editing: `test_images/women.jpg`, change outfit to vivid blue and background to snowy mountain lake.

GPU command:

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

Expected outputs:

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

Run the standard-step parameter sweep:

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

For a quick sanity check, use:

```bash
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

## Quality Checks

Before reporting a configuration as promising, inspect images and text directly.

Understanding:

```text
text output is non-empty
visible image text is at least partly read correctly
explanation matches the image
```

Text-to-image:

```text
contains raccoon-like subject
contains rocket/astronaut cues
follows the long prompt without pure noise or black image
```

Editing:

```text
source identity and pose remain recognizable
outfit is clearly blue
background resembles snowy mountain lake
white poodle applique is preferably preserved
```

Also check `topk_stats.csv`:

```text
deep layers should have lower keep ratio than shallow layers
later generation/editing timesteps should have lower keep ratio than early timesteps
active VAE latent/self KV should be protected, not counted as evicted candidates
```

## Reporting

Report at least:

```text
config name
beta
alpha_understanding
gen_keep_start -> gen_keep_end
mean_keep_ratio
theoretical_skipped_kv_gib
baseline and Uni4Uni-KV outputs
human quality note for each task
elapsed time and CUDA reserved memory
```

Do not claim speedup from this mask-probe implementation. It may be slower because TopK and masking add overhead.

## Older BR-KV Tools

The repository still contains older BR-KV calibration utilities under `brkv/`, `configs/`, and `examples/`. They are kept for reference and schedule-building experiments, but the current main path is the Uni4Uni-KV demo and sweep above.
