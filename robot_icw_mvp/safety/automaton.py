"""Safety checks for the ICW adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class SafetyCheckResult:
    safe: bool
    clamped_actions: torch.Tensor
    report: Dict[str, float]


class SafetyAutomaton:
    """Applies lightweight safety checks to candidate actions."""

    def __init__(self, torque_limit_scale: float = 1.1) -> None:
        self._torque_limit_scale = torque_limit_scale
        self._latest_report: Dict[str, float] = {}

    def _prepare_limits(
        self, control_limits
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if control_limits is None:
            return None
        upper, lower = control_limits
        if upper is None or lower is None:
            return None
        return (
            torch.as_tensor(upper),
            torch.as_tensor(lower),
        )

    def validate(self, actions: torch.Tensor, control_limits) -> SafetyCheckResult:
        limits = self._prepare_limits(control_limits)
        if limits is None:
            report = {"max_abs_action": float(actions.abs().max().item())}
            self._latest_report = report
            return SafetyCheckResult(True, actions, report)

        upper, lower = limits
        upper = upper.to(actions.device) * self._torque_limit_scale
        lower = lower.to(actions.device) * self._torque_limit_scale
        clamped = torch.max(torch.min(actions, upper), lower)
        within = torch.allclose(clamped, actions, atol=1e-6)
        report = {
            "max_abs_action": float(actions.abs().max().item()),
            "limit": float(upper.abs().max().item()),
            "within_limits": float(within),
        }
        self._latest_report = report
        return SafetyCheckResult(bool(within), clamped, report)

    @property
    def latest_report(self) -> Dict[str, float]:
        return dict(self._latest_report)

