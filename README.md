# BAGEL BR-KV Lifetime Experiments

BR-KV is a simple benefit-risk method for conditioning-KV lifetime management in unified multimodal generation models such as BAGEL.

The main idea is deliberately simple:

```text
calibrate each role/layer KV cell offline,
estimate its retired KV-attention work benefit,
measure its empirical quality risk,
then choose the earliest safe retire-after cutoff from a lookup table.
```

At inference time the policy is static. There is no online attention logging, classifier, VLM judge, or extra forward pass. The only runtime action is:

```text
if denoising_step >= tau_cell:
    hide or skip this conditioning-KV cell
```

## Method

Define a conditioning-KV cell:

```text
c = role x layer_band
```

For BAGEL editing we use:

```text
roles = main:image, main:text, cfg_text:image, cfg_img:text
layer_bands = L01-L07, L08-L14, L14-L20, L21-L23, L24-L27
cutoffs = 0, T/3, 2T/3, T+1
```

For each cell and cutoff, the calibration run generates an edited image. A human or VLM judge labels the result:

```text
pass, minor, fail
```

BR-KV computes:

```text
risk(c,tau) = average label loss or fail rate
benefit(c,tau) = retired_step_fraction * layer_count * kv_token_count * branch_weight
```

Then it selects:

```text
tau*_c = earliest tau whose risk is within the profile budget
```

This produces static policy tables:

```text
conservative_scheduler.json
balanced_scheduler.json
aggressive_scheduler.json
```

## Quick Start

```bash
python -m brkv.build_brkv_schedule \
  --calibration-csv examples/labeled_calibration_smoke.csv \
  --output-dir outputs/schedule_smoke \
  --profiles conservative:0,balanced:1,aggressive:2 \
  --fail-rate-budget 0.10
```

If you have a local BAGEL checkout with the experiment scripts installed:

```bash
scripts/install_into_bagel.sh /path/to/BAGEL

python -m brkv.launch_bagel_experiment \
  --config configs/audrey_smoke.yaml \
  --bagel-root /path/to/BAGEL \
  --stage calibration
```

## Important Caveat

The current BAGEL local scripts zero retired KV spans as a correctness probe. This validates the policy quality effect but does not prove speedup. A speed claim requires a GPU implementation that skips retired KV before concatenation or packed attention indexing.
