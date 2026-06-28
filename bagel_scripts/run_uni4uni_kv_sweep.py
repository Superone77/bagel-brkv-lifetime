#!/usr/bin/env python3
"""Sweep Uni4Uni-KV parameters for BAGEL three-task demo.

The script repeatedly invokes run_pyramid_topk_kv_demo.py, which now implements
the Uni4Uni-KV mask probe. It is intentionally orchestration-only so the model
code path stays shared with the single-run demo.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_schedules(text: str) -> list[tuple[float, float]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"schedule must look like 0.8->0.1, got {item!r}")
        start, end = item.split("->", 1)
        out.append((float(start), float(end)))
    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=root.parent / "outputs" / "bagel_uni4uni_kv_sweep")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--betas", type=parse_csv_floats, default="8,14,20")
    parser.add_argument("--alphas", type=parse_csv_ints, default="8,16")
    parser.add_argument("--schedules", type=parse_schedules, default="0.8->0.1,0.8->0.2,0.9->0.3")
    parser.add_argument("--understanding-keep-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of configs for smoke runs.")
    parser.add_argument("--skip-baseline-after-first", action="store_true", default=True)
    parser.add_argument("--demo-script", type=Path, default=root / "run_pyramid_topk_kv_demo.py")
    return parser.parse_args()


def summarize_run(summary_path: Path, config_name: str, beta: float, alpha: int, start: float, end: float) -> dict[str, str | float | int]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policy = summary.get("topk_policy_summary", {})
    row: dict[str, str | float | int] = {
        "config": config_name,
        "beta": beta,
        "alpha_understanding": alpha,
        "gen_keep_start": start,
        "gen_keep_end": end,
        "mean_keep_ratio": policy.get("mean_keep_ratio", ""),
        "total_skipped_tokens": policy.get("total_skipped_tokens", ""),
        "theoretical_skipped_kv_gib": policy.get("theoretical_skipped_kv_gib", ""),
    }
    for variant in ("baseline", "topk"):
        for task in ("understanding", "generation", "editing"):
            metrics = summary["runs"][variant][task]["metrics"]
            row[f"{variant}_{task}_elapsed_s"] = metrics["elapsed_s"]
            if task == "understanding":
                row[f"{variant}_{task}_text"] = summary["runs"][variant][task]["text"].replace("\n", " ")
            else:
                row[f"{variant}_{task}_image"] = summary["runs"][variant][task].get("image_path", "")
    return row


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.demo_script = args.demo_script.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    configs = list(itertools.product(args.betas, args.alphas, args.schedules))
    if args.limit > 0:
        configs = configs[: args.limit]

    rows = []
    for i, (beta, alpha, (start, end)) in enumerate(configs):
        config_name = f"b{beta:g}_a{alpha}_s{int(start*100):02d}_e{int(end*100):02d}"
        out_dir = args.output_root / config_name
        cmd = [
            sys.executable,
            str(args.demo_script),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--image-size",
            str(args.image_size),
            "--timesteps",
            str(args.timesteps),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--understanding-keep-ratio",
            str(args.understanding_keep_ratio),
            "--gen-keep-start",
            str(start),
            "--gen-keep-end",
            str(end),
            "--beta",
            str(beta),
            "--alpha-understanding",
            str(alpha),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(out_dir),
        ]
        print(f"\n[{i + 1}/{len(configs)}] {config_name}", flush=True)
        t0 = time.perf_counter()
        with (args.output_root / f"{config_name}.log").open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            subprocess.run(cmd, cwd=args.demo_script.parent, check=True, stdout=log, stderr=subprocess.STDOUT)
        row = summarize_run(out_dir / "summary.json", config_name, beta, alpha, start, end)
        row["wall_elapsed_s"] = time.perf_counter() - t0
        rows.append(row)

        aggregate_path = args.output_root / "sweep_summary.csv"
        with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {aggregate_path}", flush=True)


if __name__ == "__main__":
    main()
