from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


LABEL_LOSS = {
    "pass": 0.0,
    "minor": 1.0,
    "acceptable": 1.0,
    "fail": 4.0,
}

TAU_ORDER = {
    "0": 0,
    "T/4": 1,
    "T/3": 2,
    "T/2": 3,
    "2T/3": 4,
    "3T/4": 5,
    "T+1": 6,
}

LAYER_BANDS = {
    "L01-L07": tuple(range(1, 8)),
    "L08-L14": tuple(range(8, 15)),
    "L14-L20": tuple(range(14, 21)),
    "L21-L23": tuple(range(21, 24)),
    "L24-L27": tuple(range(24, 28)),
}

DEFAULT_BRANCH_WEIGHTS = {
    "main:image": 1.0,
    "main:text": 1.0,
    "cfg_text:image": 1.0,
    "cfg_img:text": 1.0,
}


@dataclass(frozen=True)
class Cell:
    task: str
    role: str
    layer_band: str

    @property
    def key(self) -> str:
        return f"{self.task}|{self.role}|{self.layer_band}"


def tau_sort_key(tau: str) -> int:
    return TAU_ORDER.get(tau, 999)


def label_loss(label: str) -> float | None:
    normalized = (label or "").strip().lower()
    if not normalized or normalized == "unlabeled":
        return None
    return LABEL_LOSS.get(normalized, 2.0)


def is_fail(label: str) -> bool | None:
    loss = label_loss(label)
    if loss is None:
        return None
    return loss >= LABEL_LOSS["fail"]


def retired_step_fraction(cutoff: str, total_steps: int | None = None) -> float:
    if cutoff == "T+1":
        return 0.0
    if cutoff == "0":
        return 1.0
    if cutoff == "T/4":
        return 0.75
    if cutoff == "T/3":
        return 2.0 / 3.0
    if cutoff == "T/2":
        return 0.5
    if cutoff == "2T/3":
        return 1.0 / 3.0
    if cutoff == "3T/4":
        return 0.25
    if total_steps is not None:
        try:
            step = int(cutoff)
        except ValueError:
            return 0.0
        return max(0.0, min(1.0, (total_steps - step) / total_steps))
    return 0.0


def layer_count(layer_band: str) -> int:
    return len(LAYER_BANDS.get(layer_band, ()))


def static_benefit(
    *,
    role: str,
    layer_band: str,
    cutoff: str,
    kv_tokens: float = 1.0,
    branch_weights: Mapping[str, float] | None = None,
) -> float:
    weights = branch_weights or DEFAULT_BRANCH_WEIGHTS
    return (
        max(0.0, float(kv_tokens))
        * layer_count(layer_band)
        * retired_step_fraction(cutoff)
        * float(weights.get(role, 1.0))
    )
