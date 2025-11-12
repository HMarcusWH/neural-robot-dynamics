"""Wisdom layer for the ICW adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from robot_icw_mvp.constants import BACKEND_NEURAL, canonicalize_backend
from robot_icw_mvp.geometry.cache import GeometrySnapshot


@dataclass
class WisdomSummary:
    metrics: Dict[str, float] = field(default_factory=dict)
    certificates: Dict[str, bool] = field(default_factory=dict)
    certificates_passed: bool = True


class WisdomModule:
    """Aggregates continuity diagnostics and simple certificates."""

    def __init__(
        self,
        disc_weight: float = 1.0,
        rupture_weight: float = 0.5,
        connectedness_max: float = 5.0,
        monotone_max: float = 5.0,
        compression_max: float = 5.0,
        residual_min: float = 0.0,
    ) -> None:
        self._disc_weight = disc_weight
        self._rupture_weight = rupture_weight
        self._connectedness_max = connectedness_max
        self._monotone_max = monotone_max
        self._compression_max = compression_max
        self._residual_min = residual_min
        self._latest_summary = WisdomSummary()

    def _evaluate_certificates(self, snapshot: GeometrySnapshot) -> WisdomSummary:
        step_norm = float(snapshot.step_norm.mean().item())
        rupture = snapshot.mean_rupture()
        disc = self._disc_weight * step_norm + self._rupture_weight * rupture
        max_rupture = float(snapshot.rupture.max().item()) if snapshot.rupture.numel() else 0.0
        residual_agreement = float(
            snapshot.metadata.get("residual_agreement", snapshot.step_norm.new_ones(snapshot.step_norm.shape)).mean().item()
        )
        certificates = {
            "C1_connectedness": disc < self._connectedness_max,
            "C2_monotone_up": max_rupture < self._monotone_max,
            "C3_compression_robust": step_norm < self._compression_max,
            "C4_residual_agreement": residual_agreement >= self._residual_min,
        }
        passed = all(certificates.values())
        metrics = {
            "disc": disc,
            "rupture": rupture,
            "step_norm": step_norm,
        }
        metrics.update(
            {
                f"certificates/{name}": 1.0 if flag else 0.0
                for name, flag in certificates.items()
            }
        )
        summary = WisdomSummary(
            metrics=metrics,
            certificates=certificates,
            certificates_passed=passed,
        )
        return summary

    def precheck(self, snapshot: GeometrySnapshot) -> WisdomSummary:
        summary = self._evaluate_certificates(snapshot)
        self._latest_summary = summary
        return summary

    def summarize(self, snapshot: GeometrySnapshot, decision_backend: str) -> WisdomSummary:
        summary = self._evaluate_certificates(snapshot)
        backend_label = canonicalize_backend(decision_backend)
        summary.metrics["backend"] = 1.0 if backend_label == BACKEND_NEURAL else 0.0
        summary.metrics["certificates_passed"] = (
            1.0 if summary.certificates_passed else 0.0
        )
        self._latest_summary = summary
        return summary

    @property
    def latest_summary(self) -> WisdomSummary:
        return self._latest_summary

