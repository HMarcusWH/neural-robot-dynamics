"""Safety checks for the ICW adapter."""

from __future__ import annotations

from typing import Dict

import torch


class SafetyAutomaton:
    """Applies lightweight safety checks to candidate actions."""

    def __init__(self, torque_limit_scale: float = 1.1) -> None:
        self._torque_limit_scale = torque_limit_scale
        self._latest_report: Dict[str, float] = {}

    def validate(self, actions: torch.Tensor, control_limits) -> bool:
        if control_limits is None:
            self._latest_report = {"max_abs_action": float(actions.abs().max().item())}
            return True
        upper, lower = control_limits
        if upper is None or lower is None:
            self._latest_report = {"max_abs_action": float(actions.abs().max().item())}
            return True
        max_allowed = torch.as_tensor(upper, device=actions.device) * self._torque_limit_scale
        min_allowed = torch.as_tensor(lower, device=actions.device) * self._torque_limit_scale
        within = torch.all(actions <= max_allowed) and torch.all(actions >= min_allowed)
        self._latest_report = {
            "max_abs_action": float(actions.abs().max().item()),
            "limit": float(max_allowed.abs().max().item()),
            "within_limits": float(within),
        }
        return bool(within)

    @property
    def latest_report(self) -> Dict[str, float]:
        return dict(self._latest_report)

