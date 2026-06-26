#!/usr/bin/env python3
"""Launch BAGEL BR-KV calibration or joint-policy validation runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bagel-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "policy", "print"), required=True)
    parser.add_argument("--scheduler-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with `pip install pyyaml`.") from exc
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def append_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
    else:
        cmd.extend([flag, str(value)])


def require_script(bagel_root: Path, script_name: str) -> Path:
    script = bagel_root / script_name
    if not script.exists():
        raise SystemExit(
            f"Missing BAGEL script: {script}\n"
            "Copy the BR-KV-compatible BAGEL experiment scripts into the BAGEL root first."
        )
    return script


def calibration_command(config: dict[str, Any], bagel_root: Path) -> list[str]:
    script = require_script(bagel_root, config.get("calibration_script", "analyze_conditioning_kv_role_cutoff_standard.py"))
    run = config["run"]
    cmd = [sys.executable, str(script)]
    for key in (
        "model_path",
        "output_dir",
        "image",
        "device",
        "dtype",
        "prompt",
        "image_size",
        "timesteps",
        "cfg_text_scale",
        "cfg_img_scale",
        "seed",
        "variants",
    ):
        append_arg(cmd, key, run.get(key))
    return cmd


def policy_command(config: dict[str, Any], bagel_root: Path, scheduler_json: Path | None) -> list[str]:
    if scheduler_json is None:
        raise SystemExit("--scheduler-json is required for --stage policy")
    script = require_script(bagel_root, config.get("policy_script", "run_kv_lifecycle_scheduler_policy.py"))
    run = config["run"]
    cmd = [sys.executable, str(script), "--scheduler-json", str(scheduler_json)]
    for key in (
        "model_path",
        "output_dir",
        "image",
        "device",
        "dtype",
        "prompt",
        "image_size",
        "timesteps",
        "cfg_text_scale",
        "cfg_img_scale",
        "seed",
        "span_mode",
    ):
        append_arg(cmd, key, run.get(key))
    return cmd


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bagel_root = args.bagel_root.resolve()
    if args.stage == "calibration":
        cmd = calibration_command(config, bagel_root)
    elif args.stage == "policy":
        cmd = policy_command(config, bagel_root, args.scheduler_json)
    else:
        cmd = calibration_command(config, bagel_root)

    print(json.dumps({"cwd": str(bagel_root), "cmd": cmd}, indent=2, ensure_ascii=False))
    if args.dry_run or args.stage == "print":
        return

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    subprocess.run(cmd, cwd=str(bagel_root), env=env, check=True)


if __name__ == "__main__":
    main()
