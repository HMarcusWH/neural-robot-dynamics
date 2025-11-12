"""Fold-aware zoom controller used by the ICW adapter."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import torch

from robot_icw_mvp.constants import BACKEND_ANALYTIC, BACKEND_NEURAL, BACKEND_ABSTAIN, canonicalize_backend
from robot_icw_mvp.geometry.cache import GeometrySnapshot


@dataclass
class IntuitionDecision:
    backend: str
    dt_scale: float
    fallback_backend: str = BACKEND_ANALYTIC
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def effective_backend(self) -> str:
        backend = canonicalize_backend(self.backend)
        if backend == BACKEND_ABSTAIN:
            return self.fallback_backend
        return backend


class IntuitionController:
    """Implements a lightweight version of the fold-aware zoom policy."""

    def __init__(
        self,
        hysteresis_lo: float = 0.08,
        hysteresis_hi: float = 0.15,
        abstain_quantile: float = 0.95,
        residual_window: int = 1024,
        default_backend: str = BACKEND_ANALYTIC,
        abstain_dt_scale: float = 0.5,
    ) -> None:
        if hysteresis_lo >= hysteresis_hi:
            raise ValueError("`hysteresis_lo` must be strictly smaller than `hysteresis_hi`")
        self._tau_lo = hysteresis_lo
        self._tau_hi = hysteresis_hi
        self._abstain_quantile = abstain_quantile
        self._residuals: Deque[float] = deque(maxlen=residual_window)
        self._current_backend = canonicalize_backend(default_backend)
        self._abstain_dt_scale = float(abstain_dt_scale)
        self._latest_metrics: Dict[str, float] = {}
        self._latest_decision: Optional[IntuitionDecision] = None

    @property
    def current_backend(self) -> str:
        return self._current_backend

    @property
    def abstain_dt_scale(self) -> float:
        return self._abstain_dt_scale

    def reset(self, backend: Optional[str] = None) -> None:
        self._residuals.clear()
        if backend is not None:
            self._current_backend = canonicalize_backend(backend)
        self._latest_metrics = {}
        self._latest_decision = None

    def decide(self, snapshot: GeometrySnapshot) -> IntuitionDecision:
        mean_rupture = snapshot.mean_rupture()
        step_residual = float(snapshot.step_norm.mean().item())
        self._latest_metrics = {
            "rupture": mean_rupture,
            "step_norm": step_residual,
        }

        if len(self._residuals) >= 10:
            sorted_res = sorted(self._residuals)
            idx = min(int(len(sorted_res) * self._abstain_quantile), len(sorted_res) - 1)
            threshold = sorted_res[idx]
            abstain = step_residual > threshold
        else:
            threshold = float("inf")
            abstain = False
        backend = self._current_backend
        if mean_rupture > self._tau_hi:
            backend = BACKEND_NEURAL
        elif mean_rupture < self._tau_lo:
            backend = BACKEND_ANALYTIC

        label = BACKEND_ABSTAIN if abstain else backend
        dt_scale = self._abstain_dt_scale if abstain else 1.0
        decision = IntuitionDecision(
            backend=label,
            dt_scale=dt_scale,
            fallback_backend=BACKEND_ANALYTIC,
            metrics={
                "rupture": mean_rupture,
                "step_norm": step_residual,
                "abstain_threshold": threshold,
            },
        )
        self._latest_decision = decision
        return decision

    def record(self, backend: str, snapshot: GeometrySnapshot) -> None:
        self._current_backend = canonicalize_backend(backend)
        residual = float(snapshot.step_norm.mean().item())
        if not torch.isnan(torch.tensor(residual)) and not torch.isinf(torch.tensor(residual)):
            self._residuals.append(residual)

    def latest_metrics(self) -> Dict[str, float]:
        return dict(self._latest_metrics)

    def latest_decision(self) -> Optional[IntuitionDecision]:
        return self._latest_decision

