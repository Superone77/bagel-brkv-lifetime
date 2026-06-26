#!/usr/bin/env python3
"""Build static BR-KV retirement schedules from labeled calibration rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .metrics import Cell, label_loss, is_fail, static_benefit, tau_sort_key


@dataclass(frozen=True)
class ActionStats:
    cell: Cell
    tau: str
    avg_label_loss: float
    fail_rate: float
    benefit: float
    zeroed_bytes_upper_bound: float
    mae_mean: float | None
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default="editing")
    parser.add_argument("--profiles", default="conservative:0,balanced:1,aggressive:2")
    parser.add_argument("--fail-rate-budget", type=float, default=0.10)
    parser.add_argument("--allow-unlabeled", action="store_true")
    parser.add_argument("--suffix-safe", action="store_true")
    return parser.parse_args()


def parse_profiles(text: str) -> list[tuple[str, float]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            name, budget = item.split(":", 1)
            out.append((name.strip(), float(budget)))
    return out


def read_rows(path: Path, task_name: str, allow_unlabeled: bool) -> dict[Cell, dict[str, ActionStats]]:
    grouped: dict[tuple[Cell, str], list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            role = (row.get("role") or "").strip()
            if not role or role == "baseline":
                continue
            tau = (row.get("cutoff") or row.get("retire_tau") or "").strip()
            if not tau or tau == "none":
                continue
            loss = label_loss(row.get("human_label", ""))
            fail = is_fail(row.get("human_label", ""))
            if loss is None or fail is None:
                if not allow_unlabeled:
                    continue
                loss, fail = 0.0, False
            cell = Cell(row.get("task") or task_name, role, (row.get("layer_band") or "").strip())
            zeroed = float(row.get("zeroed_bytes_upper_bound") or row.get("retired_value") or 0.0)
            mae_text = row.get("mae_vs_baseline") or row.get("mae_mean") or ""
            mae = float(mae_text) if mae_text else float("nan")
            benefit = zeroed if zeroed > 0 else static_benefit(
                role=cell.role,
                layer_band=cell.layer_band,
                cutoff=tau,
                kv_tokens=float(row.get("token_count") or 1.0),
            )
            grouped[(cell, tau)].append({"loss": float(loss), "fail": 1.0 if fail else 0.0, "benefit": benefit, "zeroed": zeroed, "mae": mae})

    by_cell: dict[Cell, dict[str, ActionStats]] = defaultdict(dict)
    for (cell, tau), rows in grouped.items():
        n = len(rows)
        maes = [r["mae"] for r in rows if r["mae"] == r["mae"]]
        by_cell[cell][tau] = ActionStats(
            cell=cell,
            tau=tau,
            avg_label_loss=sum(r["loss"] for r in rows) / n,
            fail_rate=sum(r["fail"] for r in rows) / n,
            benefit=sum(r["benefit"] for r in rows) / n,
            zeroed_bytes_upper_bound=sum(r["zeroed"] for r in rows) / n,
            mae_mean=(sum(maes) / len(maes)) if maes else None,
            n=n,
        )
    return dict(by_cell)


def is_safe(action: ActionStats, quality_budget: float, fail_rate_budget: float) -> bool:
    return action.avg_label_loss <= quality_budget and action.fail_rate <= fail_rate_budget


def suffix_safe(actions: dict[str, ActionStats], tau: str, quality_budget: float, fail_rate_budget: float) -> bool:
    return all(
        is_safe(actions[item], quality_budget, fail_rate_budget)
        for item in actions
        if tau_sort_key(item) >= tau_sort_key(tau)
    )


def choose(actions: dict[str, ActionStats], quality_budget: float, fail_rate_budget: float, use_suffix_safe: bool) -> ActionStats:
    safe = []
    for action in sorted(actions.values(), key=lambda a: tau_sort_key(a.tau)):
        if is_safe(action, quality_budget, fail_rate_budget) and (
            not use_suffix_safe or suffix_safe(actions, action.tau, quality_budget, fail_rate_budget)
        ):
            safe.append(action)
    if safe:
        return safe[0]
    return actions.get("T+1") or min(actions.values(), key=lambda a: (a.avg_label_loss, a.fail_rate, -tau_sort_key(a.tau)))


def write_profile(output_dir: Path, profile: str, budget: float, by_cell: dict[Cell, dict[str, ActionStats]], args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for cell in sorted(by_cell, key=lambda c: (c.task, c.role, c.layer_band)):
        action = choose(by_cell[cell], budget, args.fail_rate_budget, args.suffix_safe)
        rows.append({
            "task": cell.task,
            "role": cell.role,
            "layer_band": cell.layer_band,
            "retire_tau": action.tau,
            "quality_loss": action.avg_label_loss,
            "fail_rate": action.fail_rate,
            "benefit": action.benefit,
            "zeroed_bytes_upper_bound": action.zeroed_bytes_upper_bound,
            "mae_mean": action.mae_mean,
            "n": action.n,
        })
    metrics = {
        "quality_budget": budget,
        "fail_rate_budget": args.fail_rate_budget,
        "total_benefit": sum(float(r["benefit"]) for r in rows),
        "avg_quality_loss": sum(float(r["quality_loss"]) for r in rows) / max(1, len(rows)),
        "avg_fail_rate": sum(float(r["fail_rate"]) for r in rows) / max(1, len(rows)),
        "cell_count": len(rows),
    }
    payload = {"method": "BR-KV static benefit-risk conditioning-KV retirement", "profile": profile, "metrics": metrics, "scheduler": rows}
    (output_dir / f"{profile}_scheduler.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / f"{profile}_scheduler.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_cell = read_rows(args.calibration_csv, args.task_name, args.allow_unlabeled)
    if not by_cell:
        raise SystemExit("No labeled calibration rows found. Fill human_label or pass --allow-unlabeled.")
    payloads = [write_profile(args.output_dir, p, b, by_cell, args) for p, b in parse_profiles(args.profiles)]
    lines = ["# BR-KV Schedule Summary", "", "| Profile | Cells | Total benefit | Avg quality loss | Avg fail rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for p in payloads:
        m = p["metrics"]
        lines.append(f"| {p['profile']} | {m['cell_count']} | {m['total_benefit']:.1f} | {m['avg_quality_loss']:.3f} | {m['avg_fail_rate']:.3f} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps([p["metrics"] for p in payloads], indent=2))


if __name__ == "__main__":
    main()
