"""Geometry utilities for the ICW adapter.

The helpers in this module are intentionally lightweight – they build
per-rung cosine diagnostics directly from the generalized coordinates that
`NeuralEnvironment` already exposes.  The goal is not to reproduce the full
Hilbert ladder stack from the research notes, but to provide a deterministic
and well-documented approximation that the controllers can rely on when
running inside the existing NeRD pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class GeometrySnapshot:
    """Snapshot of the ladder diagnostics for a batch of environments."""

    states: torch.Tensor
    prev_states: Optional[torch.Tensor]
    ddp: torch.Tensor
    rupture: torch.Tensor
    step_norm: torch.Tensor
    rung_dims: Sequence[int]
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)
    step_index: int = 0

    def mean_rupture(self) -> float:
        return float(self.rupture.mean().item())


class LadderGeometryCache:
    """Compute and cache ladder diagnostics directly from env states."""

    def __init__(
        self,
        rung_dims: Iterable[int],
        device: torch.device,
        eps: float = 1e-6,
    ) -> None:
        self._rung_dims = tuple(sorted(set(int(d) for d in rung_dims)))
        if len(self._rung_dims) == 0:
            raise ValueError("At least one rung dimension must be provided")
        self._device = device
        self._eps = eps
        self._prev_states: Optional[torch.Tensor] = None
        self._prev_ddp: Optional[torch.Tensor] = None
        self._latest_snapshot: Optional[GeometrySnapshot] = None
        self._step_counter = 0

    @property
    def rung_dims(self) -> Sequence[int]:
        return self._rung_dims

    @property
    def latest_snapshot(self) -> Optional[GeometrySnapshot]:
        return self._latest_snapshot

    def reset(self) -> None:
        self._prev_states = None
        self._prev_ddp = None
        self._latest_snapshot = None
        self._step_counter = 0

    def _compute_ddp(
        self,
        states: torch.Tensor,
        prev_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, dim = states.shape
        ddp = torch.zeros((batch, len(self._rung_dims)), device=states.device)
        if prev_states is None:
            return ddp

        for idx, rung in enumerate(self._rung_dims):
            use_dim = min(rung, dim)
            if use_dim <= 0:
                continue
            current = states[:, :use_dim]
            previous = prev_states[:, :use_dim]
            cos = F.cosine_similarity(current, previous, dim=-1, eps=self._eps)
            ddp[:, idx] = 1.0 - cos
        return ddp

    def _compute_rupture(
        self, ddp: torch.Tensor, prev_ddp: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if prev_ddp is None:
            return ddp.norm(p=2, dim=-1)
        return torch.sum(torch.abs(ddp - prev_ddp), dim=-1)

    def observe(self, states: torch.Tensor) -> GeometrySnapshot:
        """Commit the current states and return an updated snapshot."""

        states = states.detach().to(self._device)
        prev_states = None if self._prev_states is None else self._prev_states.detach()
        ddp = self._compute_ddp(states, prev_states)
        rupture = self._compute_rupture(ddp, self._prev_ddp)
        step_norm = (
            torch.zeros(states.shape[0], device=states.device)
            if prev_states is None
            else torch.linalg.norm(states - prev_states, dim=-1)
        )
        self._step_counter += 1
        snapshot = GeometrySnapshot(
            states=states,
            prev_states=prev_states,
            ddp=ddp,
            rupture=rupture,
            step_norm=step_norm,
            rung_dims=self._rung_dims,
            metadata={"ddp": ddp, "step_norm": step_norm},
            step_index=self._step_counter,
        )
        self._prev_states = states
        self._prev_ddp = ddp
        self._latest_snapshot = snapshot
        return snapshot

    def peek(self) -> Optional[GeometrySnapshot]:
        """Return the most recent snapshot without updating the cache."""

        return self._latest_snapshot

