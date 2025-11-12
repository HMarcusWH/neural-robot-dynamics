"""Creativity controller for the ICW adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from robot_icw_mvp.geometry.cache import GeometrySnapshot


@dataclass
class CreativityProposal:
    delta_actions: Optional[torch.Tensor]
    metadata: Dict[str, float] = field(default_factory=dict)
    triggered: bool = False


class CreativityPlanner:
    """Generate merge-stable action snippets when rupture spikes."""

    def __init__(
        self,
        max_branches: int = 2,
        rupture_threshold: float = 0.2,
    ) -> None:
        self._max_branches = max_branches
        self._rupture_threshold = rupture_threshold
        self._running_branch_budget = max_branches

    def reset(self) -> None:
        self._running_branch_budget = self._max_branches

    def propose(
        self,
        snapshot: GeometrySnapshot,
        current_backend: str,
        actions: torch.Tensor,
    ) -> CreativityProposal:
        mean_rupture = snapshot.mean_rupture()
        should_branch = (
            current_backend == "neural"
            and mean_rupture > self._rupture_threshold
            and self._running_branch_budget > 0
        )
        if not should_branch:
            return CreativityProposal(delta_actions=None, metadata={"rupture": mean_rupture}, triggered=False)

        # The MVP planner takes a conservative stance: it returns a zero delta
        # but logs that a creative opportunity was detected.  Downstream users
        # can replace this with richer snippet generation without touching the
        # rest of the adapter.
        self._running_branch_budget -= 1
        zero_delta = torch.zeros_like(actions)
        return CreativityProposal(
            delta_actions=zero_delta,
            metadata={"rupture": mean_rupture},
            triggered=True,
        )

