"""Turns scores into a pass/fail decision: absolute thresholds plus no regression vs a baseline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    minimum: dict[str, float]
    maximum: dict[str, float]
    max_regression: float


def load_thresholds(path: Path) -> Thresholds:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return Thresholds(
        minimum={k: float(v) for k, v in data.get("minimum", {}).items()},
        maximum={k: float(v) for k, v in data.get("maximum", {}).items()},
        max_regression=float(data.get("regression", {}).get("max_drop", 0.0)),
    )


def check(
    scores: dict[str, float], thresholds: Thresholds, baseline: dict[str, float] | None = None
) -> list[str]:
    failures: list[str] = []
    for name, floor in thresholds.minimum.items():
        if name not in scores:
            failures.append(f"{name}: metric missing from run")
        elif scores[name] < floor:
            failures.append(f"{name}: {scores[name]:.3f} is below the minimum {floor:.3f}")
    for name, ceiling in thresholds.maximum.items():
        if name not in scores:
            failures.append(f"{name}: metric missing from run")
        elif scores[name] > ceiling:
            failures.append(f"{name}: {scores[name]:.3f} is above the maximum {ceiling:.3f}")
    if baseline is not None:
        for name in thresholds.minimum:
            if name in scores and name in baseline:
                drop = baseline[name] - scores[name]
                if drop > thresholds.max_regression + 1e-9:
                    failures.append(
                        f"{name}: dropped {drop:.3f} vs baseline ({baseline[name]:.3f} -> {scores[name]:.3f}),"
                        f" allowed {thresholds.max_regression:.3f}"
                    )
    return failures
