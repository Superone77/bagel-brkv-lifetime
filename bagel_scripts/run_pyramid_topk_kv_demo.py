#!/usr/bin/env python3
"""Run BAGEL Uni4Uni-KV probes.

This is a correctness/quality probe.  The TopK path masks non-selected keys in
attention logits; it reports theoretical skipped KV work but does not physically
compact K/V tensors or claim wall-clock speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from demo_mps import build_model, choose_device, choose_dtype, ensure_model
from inferencer import InterleaveInferencer
from modeling.bagel.qwen2_navit import clear_topk_kv_policy, set_topk_kv_policy
from modeling.qwen2 import Qwen2Tokenizer


UNDERSTANDING_PROMPT = "Read all visible text in the image and explain what the image is about."

GENERATION_PROMPT = (
    "Create a cinematic square illustration of a clever rocket-powered raccoon astronaut launching from a "
    "moonlit forest clearing toward a field of bright stars. The character wears a compact silver space suit, "
    "transparent bubble helmet, tiny mission patches, and a glowing jetpack with warm orange exhaust. Around the "
    "launch site are pine trees, scattered tools, a hand-drawn star map, and soft blue smoke lit by the rocket plume. "
    "Make the raccoon expressive and heroic, with sharp whiskers, striped tail, bright curious eyes, and one paw "
    "holding a small navigation tablet. The scene should feel adventurous, whimsical, richly detailed, and polished, "
    "with dramatic rim lighting, crisp fur texture, realistic metal reflections, subtle sparks, and a deep night sky. "
    "Avoid text, logos, watermarks, distorted anatomy, extra limbs, or cropped face."
)

EDIT_PROMPT = (
    "Edit the input portrait so that the person's red outfit becomes a vivid bright blue coat, and replace the gray "
    "studio background with a realistic snowy mountain lake landscape. Preserve the same person identity, facial "
    "expression, hairstyle, pose, hand position, framing, lighting, and the white poodle applique on the outfit. "
    "Keep the subject sharp and natural. Avoid changing the face shape, camera angle, body pose, or composition."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=root.parent / "models" / "BAGEL-7B-MoT")
    parser.add_argument("--output-dir", type=Path, default=root.parent / "outputs" / "bagel_pyramid_topk_kv_demo")
    parser.add_argument("--understanding-image", type=Path, default=root / "test_images" / "meme.jpg")
    parser.add_argument("--edit-image", type=Path, default=root / "test_images" / "women.jpg")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--understanding-keep-ratio", type=float, default=0.5)
    parser.add_argument("--gen-keep-start", type=float, default=0.8)
    parser.add_argument("--gen-keep-end", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--alpha-understanding", type=int, default=8)
    parser.add_argument("--layer-decay-strength", type=float, default=None)
    parser.add_argument("--min-keep", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot() -> dict[str, float | int]:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_gib = max_rss / 1024**3 if platform.system() == "Darwin" else max_rss / 1024**2
    snap: dict[str, float | int] = {
        "pid": os.getpid(),
        "max_rss_gib": max_rss_gib,
        "mps_current_gib": 0.0,
        "mps_driver_gib": 0.0,
        "cuda_current_gib": 0.0,
        "cuda_reserved_gib": 0.0,
    }
    if torch.cuda.is_available():
        snap["cuda_current_gib"] = torch.cuda.memory_allocated() / 1024**3
        snap["cuda_reserved_gib"] = torch.cuda.memory_reserved() / 1024**3
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        if current is not None:
            snap["mps_current_gib"] = current() / 1024**3
        if driver is not None:
            snap["mps_driver_gib"] = driver() / 1024**3
    return snap


def timed(name: str, device: torch.device, fn):
    sync(device)
    before = memory_snapshot()
    start = time.perf_counter()
    result = fn()
    sync(device)
    elapsed = time.perf_counter() - start
    after = memory_snapshot()
    metrics = {
        "elapsed_s": elapsed,
        "memory_before": before,
        "memory_after": after,
        "mps_driver_delta_gib": float(after["mps_driver_gib"]) - float(before["mps_driver_gib"]),
        "cuda_reserved_delta_gib": float(after["cuda_reserved_gib"]) - float(before["cuda_reserved_gib"]),
        "max_rss_delta_gib": float(after["max_rss_gib"]) - float(before["max_rss_gib"]),
    }
    print(f"{name}: {elapsed:.2f}s", flush=True)
    return result, metrics


@dataclass
class PyramidTopKPolicy:
    total_layers: int
    understanding_keep_ratio: float
    gen_keep_start: float
    gen_keep_end: float
    beta: float
    alpha_understanding: int
    bytes_per_kv_token_layer: int
    min_keep: int = 8
    enabled: bool = True
    task: str = ""
    variant: str = "topk"
    phase: str = "idle"
    step_index: int = 0
    total_steps: int = 1
    rows: list[dict[str, Any]] = field(default_factory=list)

    def set_task(self, task: str) -> None:
        self.task = task

    def set_context(self, *, phase=None, step_index=None, total_steps=None) -> None:
        if phase is not None:
            self.phase = str(phase)
        if step_index is not None:
            self.step_index = int(step_index)
        if total_steps is not None:
            self.total_steps = max(1, int(total_steps))

    def should_apply(self, layer_idx: int, mode: str | None) -> bool:
        return True

    def extra_protected_key_mask(
        self,
        *,
        key_len: int,
        protected_mask: torch.Tensor,
        layer_idx: int,
        mode: str | None,
        device,
    ):
        if self.phase != "understanding" or self.alpha_understanding <= 0 or key_len <= 0:
            return None
        mask = torch.zeros(key_len, device=device, dtype=torch.bool)
        unprotected = torch.nonzero(~protected_mask, as_tuple=False).flatten()
        if unprotected.numel() == 0:
            return mask
        keep = unprotected[-min(self.alpha_understanding, int(unprotected.numel())) :]
        mask[keep.long()] = True
        return mask

    def _base_ratio(self) -> float:
        if self.phase == "understanding":
            return self.understanding_keep_ratio
        denom = max(1, self.total_steps - 1)
        progress = min(1.0, max(0.0, self.step_index / denom))
        return self.gen_keep_start + (self.gen_keep_end - self.gen_keep_start) * progress

    def _layer_ratio_multiplier(self, layer_idx: int) -> float:
        beta = max(float(self.beta), 1.0)
        layers = max(1, int(self.total_layers))
        top = 1.0 / beta
        bottom = 2.0 - top
        if layers == 1:
            return 1.0
        layer_progress = min(1.0, max(0.0, layer_idx / (layers - 1)))
        return bottom - (bottom - top) * layer_progress

    def keep_ratio(self, layer_idx: int, mode: str | None) -> float:
        base = min(1.0, max(0.0, self._base_ratio()))
        return min(1.0, max(0.0, base * self._layer_ratio_multiplier(layer_idx)))

    def record(
        self,
        *,
        layer_idx: int,
        mode: str,
        sample_idx: int,
        query_len: int,
        key_len: int,
        protected_key_len: int = 0,
        candidate_key_len: int | None = None,
        keep_candidate_k: int | None = None,
        keep_k: int,
        keep_ratio: float,
        candidate_keep_ratio: float | None = None,
        target_keep_ratio: float,
        skipped_tokens: int,
    ) -> None:
        if candidate_key_len is None:
            candidate_key_len = key_len
        if keep_candidate_k is None:
            keep_candidate_k = keep_k
        if candidate_keep_ratio is None:
            candidate_keep_ratio = keep_candidate_k / max(candidate_key_len, 1)
        self.rows.append(
            {
                "variant": self.variant,
                "task": self.task,
                "phase": self.phase,
                "mode": mode,
                "step_index": self.step_index,
                "total_steps": self.total_steps,
                "layer": layer_idx,
                "sample_idx": sample_idx,
                "query_len": query_len,
                "key_len": key_len,
                "protected_key_len": protected_key_len,
                "candidate_key_len": candidate_key_len,
                "keep_k": keep_k,
                "keep_candidate_k": keep_candidate_k,
                "keep_ratio": keep_ratio,
                "candidate_keep_ratio": candidate_keep_ratio,
                "target_keep_ratio": target_keep_ratio,
                "skipped_tokens": skipped_tokens,
                "theoretical_skipped_kv_bytes": skipped_tokens * self.bytes_per_kv_token_layer,
            }
        )

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"rows": 0}
        skipped = sum(int(row["skipped_tokens"]) for row in self.rows)
        total = sum(int(row["key_len"]) for row in self.rows)
        return {
            "rows": len(self.rows),
            "total_key_tokens_seen": total,
            "total_skipped_tokens": skipped,
            "mean_keep_ratio": 1.0 - skipped / max(total, 1),
            "theoretical_skipped_kv_gib": sum(int(row["theoretical_skipped_kv_bytes"]) for row in self.rows) / 1024**3,
        }


def write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_understanding(inferencer: InterleaveInferencer, image_path: Path, max_new_tokens: int):
    image = pil_img2rgb(Image.open(image_path))
    return inferencer(
        image=image,
        text=UNDERSTANDING_PROMPT,
        understanding_output=True,
        do_sample=False,
        text_temperature=0.3,
        max_think_token_n=max_new_tokens,
    )


def run_generation(inferencer: InterleaveInferencer, image_size: int, timesteps: int, cfg_text_scale: float):
    return inferencer(
        text=GENERATION_PROMPT,
        think=False,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(image_size, image_size),
    )


def run_editing(
    inferencer: InterleaveInferencer,
    image_path: Path,
    timesteps: int,
    cfg_text_scale: float,
    cfg_img_scale: float,
):
    image = pil_img2rgb(Image.open(image_path))
    return inferencer(
        image=image,
        text=EDIT_PROMPT,
        think=False,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BAGEL Uni4Uni-KV Demo",
        "",
        "This is a mask-based correctness probe. It does not physically compact KV tensors and should not be read as a speedup claim.",
        "",
        "Uni4Uni-KV protects current latent/self KV, optionally protects recent text-decoding tokens via alpha, and applies a beta-shaped PyramidKV layer budget to conditioning/past KV candidates.",
        "",
        "## Prompts",
        "",
        f"- Understanding image: `{summary['inputs']['understanding_image']}`",
        f"- Understanding prompt: {UNDERSTANDING_PROMPT}",
        f"- Generation prompt: {GENERATION_PROMPT}",
        f"- Editing image: `{summary['inputs']['edit_image']}`",
        f"- Editing prompt: {EDIT_PROMPT}",
        "",
        "## Outputs",
        "",
    ]
    for variant in ("baseline", "topk"):
        lines.extend(
            [
                f"### {variant}",
                "",
                f"- Understanding text: {summary['runs'][variant]['understanding']['text']}",
                f"- Generation image: `{summary['runs'][variant]['generation'].get('image_path', '')}`",
                f"- Editing image: `{summary['runs'][variant]['editing'].get('image_path', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## TopK Summary",
            "",
            "```json",
            json.dumps(summary.get("topk_policy_summary", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Human Observation",
            "",
            "- Understanding: TODO",
            "- Text-to-image: TODO",
            "- Editing: TODO",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_topk_kv_policy()

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    model_path = ensure_model(args.model_path, skip_download=True)

    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"output_dir: {args.output_dir}")

    set_seed(args.seed)
    model, vae_model = build_model(model_path, device, dtype)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=ImageTransform(args.image_size, min(args.image_size, 512), 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )

    llm_config = model.config.llm_config
    bytes_per_kv_token_layer = 2 * llm_config.num_key_value_heads * (llm_config.hidden_size // llm_config.num_attention_heads) * 2
    policy = PyramidTopKPolicy(
        total_layers=llm_config.num_hidden_layers,
        understanding_keep_ratio=args.understanding_keep_ratio,
        gen_keep_start=args.gen_keep_start,
        gen_keep_end=args.gen_keep_end,
        beta=args.beta,
        alpha_understanding=args.alpha_understanding,
        bytes_per_kv_token_layer=bytes_per_kv_token_layer,
        min_keep=args.min_keep,
    )

    summary: dict[str, Any] = {
        "method": "Uni4Uni-KV mask probe",
        "device": str(device),
        "dtype": str(dtype),
        "inputs": {
            "understanding_image": str(args.understanding_image),
            "edit_image": str(args.edit_image),
        },
        "config": {
            "image_size": args.image_size,
            "timesteps": args.timesteps,
            "max_new_tokens": args.max_new_tokens,
            "understanding_keep_ratio": args.understanding_keep_ratio,
            "gen_keep_start": args.gen_keep_start,
            "gen_keep_end": args.gen_keep_end,
            "beta": args.beta,
            "alpha_understanding": args.alpha_understanding,
            "layer_decay_strength_legacy": args.layer_decay_strength,
            "min_keep": args.min_keep,
            "seed": args.seed,
            "bytes_per_kv_token_layer": bytes_per_kv_token_layer,
        },
        "runs": {"baseline": {}, "topk": {}},
    }

    for variant in ("baseline", "topk"):
        print(f"\n## Variant: {variant}", flush=True)
        if variant == "baseline":
            clear_topk_kv_policy()
        else:
            policy.rows.clear()
            policy.enabled = True
            set_topk_kv_policy(policy)

        for task_name, runner in (
            ("understanding", lambda: run_understanding(inferencer, args.understanding_image, args.max_new_tokens)),
            ("generation", lambda: run_generation(inferencer, args.image_size, args.timesteps, args.cfg_text_scale)),
            ("editing", lambda: run_editing(inferencer, args.edit_image, args.timesteps, args.cfg_text_scale, args.cfg_img_scale)),
        ):
            set_seed(args.seed)
            if variant == "topk":
                policy.set_task(task_name)
                policy.set_context(phase="idle", step_index=0, total_steps=1)
            result, metrics = timed(f"{variant}/{task_name}", device, runner)
            payload: dict[str, Any] = {"metrics": metrics}
            if task_name == "understanding":
                payload["text"] = result["text"]
            else:
                image_path = args.output_dir / f"{variant}_{task_name}.png"
                result["image"].save(image_path)
                payload["image_path"] = str(image_path)
            summary["runs"][variant][task_name] = payload

    clear_topk_kv_policy()
    stats_path = args.output_dir / "topk_stats.csv"
    write_stats_csv(stats_path, policy.rows)
    summary["topk_stats_csv"] = str(stats_path)
    summary["topk_policy_summary"] = policy.summary()
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "run_report.md", summary)
    print(f"summary: {summary_path}")
    print(f"topk_stats: {stats_path}")


if __name__ == "__main__":
    main()
