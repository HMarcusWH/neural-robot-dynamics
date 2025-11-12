"""Wisdom layer for the ICW adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from robot_icw_mvp.geometry.cache import GeometrySnapshot


@dataclass
class WisdomSummary:
    metrics: Dict[str, float] = field(default_factory=dict)
    certificates_passed: bool = True


class WisdomModule:
    """Aggregates continuity diagnostics and simple certificates."""

    def __init__(self, disc_weight: float = 1.0, rupture_weight: float = 0.5) -> None:
        self._disc_weight = disc_weight
        self._rupture_weight = rupture_weight
        self._latest_summary = WisdomSummary()

    def summarize(self, snapshot: GeometrySnapshot, decision_backend: str) -> WisdomSummary:
        step_norm = float(snapshot.step_norm.mean().item())
        rupture = snapshot.mean_rupture()
        disc = self._disc_weight * step_norm + self._rupture_weight * rupture
        certificates = disc < 5.0  # lightweight continuity gate
        self._latest_summary = WisdomSummary(
            metrics={
                "disc": disc,
                "rupture": rupture,
                "step_norm": step_norm,
                "backend": 1.0 if decision_backend == "neural" else 0.0,
            },
            certificates_passed=certificates,
        )
        return self._latest_summary

    @property
    def latest_summary(self) -> WisdomSummary:
        return self._latest_summary

