"""Adapter that injects the ICW controllers into ``NeuralEnvironment``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

from robot_icw_mvp.creativity.planner import CreativityPlanner, CreativityProposal
from robot_icw_mvp.geometry.cache import LadderGeometryCache, GeometrySnapshot
from robot_icw_mvp.intuition.zoom_controller import IntuitionController, IntuitionDecision
from robot_icw_mvp.safety.automaton import SafetyAutomaton
from robot_icw_mvp.wisdom.memory_lenses import WisdomModule, WisdomSummary

DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "configs" / "default_icw.yaml"


@dataclass
class ICWDecision:
    backend: str
    action_delta: Optional[torch.Tensor] = None
    dt_scale: float = 1.0
    abstain: bool = False
    fallback_backend: str = "ground-truth"
    extras: Dict[str, float] = field(default_factory=dict)
    creativity: CreativityProposal = field(default=None)
    intuition: Optional[IntuitionDecision] = None
    wisdom: Optional[WisdomSummary] = None

    @property
    def backend_to_apply(self) -> str:
        if self.abstain:
            return self.fallback_backend
        return self.backend


class ICWController:
    """High level orchestrator for the Intuition/Creativity/Wisdom modules."""

    def __init__(
        self,
        env,
        geometry: LadderGeometryCache,
        intuition: IntuitionController,
        creativity: CreativityPlanner,
        wisdom: WisdomModule,
        safety: SafetyAutomaton,
    ) -> None:
        self._env = env
        self._geometry = geometry
        self._intuition = intuition
        self._creativity = creativity
        self._wisdom = wisdom
        self._safety = safety
        self._latest_decision: Optional[ICWDecision] = None
        self._latest_snapshot: Optional[GeometrySnapshot] = None

        # Prime the geometry cache with the environment's current state so
        # that the first decision has a defined reference point.
        if env.states is not None:
            self._geometry.observe(env.states)
            self._intuition.reset(self._intuition.current_backend)

    @staticmethod
    def _as_float(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    @property
    def current_backend(self) -> str:
        decision = self._latest_decision
        if decision is None:
            return self._intuition.current_backend
        return decision.backend_to_apply

    def before_step(self, actions: torch.Tensor) -> ICWDecision:
        snapshot = self._geometry.peek()
        if snapshot is None:
            snapshot = self._geometry.observe(self._env.states)
        intuition_decision = self._intuition.decide(snapshot)
        creativity_proposal = self._creativity.propose(
            snapshot, intuition_decision.backend, actions
        )

        if creativity_proposal.delta_actions is not None:
            candidate_actions = actions + creativity_proposal.delta_actions
        else:
            candidate_actions = actions

        safe = self._safety.validate(candidate_actions, self._env.control_limits)
        abstain = intuition_decision.abstain or not safe
        backend = intuition_decision.backend
        safety_report = self._safety.latest_report
        extras = {
            "intuition/rupture": self._as_float(intuition_decision.metrics.get("rupture", 0.0)),
            "intuition/step_norm": self._as_float(intuition_decision.metrics.get("step_norm", 0.0)),
            "intuition/abstain_threshold": self._as_float(
                intuition_decision.metrics.get("abstain_threshold", float("inf"))
            ),
            "creativity/triggered": 1.0 if creativity_proposal.triggered else 0.0,
            "creativity/rupture": self._as_float(
                creativity_proposal.metadata.get("rupture", intuition_decision.metrics.get("rupture", 0.0))
            ),
            "intuition/dt_scale": self._as_float(intuition_decision.dt_scale),
        }
        for key, value in safety_report.items():
            extras[f"safety/{key}"] = self._as_float(value)
        decision = ICWDecision(
            backend=backend,
            action_delta=creativity_proposal.delta_actions,
            dt_scale=intuition_decision.dt_scale,
            abstain=abstain,
            fallback_backend="ground-truth",
            extras=extras,
            creativity=creativity_proposal,
            intuition=intuition_decision,
        )
        self._latest_decision = decision
        self._latest_snapshot = snapshot
        return decision

    def after_step(self, states: torch.Tensor, backend: str) -> None:
        snapshot = self._geometry.observe(states)
        self._intuition.record(backend, snapshot)
        wisdom_summary = self._wisdom.summarize(snapshot, backend)
        if self._latest_decision is not None:
            self._latest_decision.wisdom = wisdom_summary
            self._latest_decision.extras.update(
                {f"wisdom/{k}": self._as_float(v) for k, v in wisdom_summary.metrics.items()}
            )
            self._latest_decision.extras["wisdom/certificates"] = (
                1.0 if wisdom_summary.certificates_passed else 0.0
            )
        self._latest_snapshot = snapshot

    def populate_extras(self, extras: Dict[str, float]) -> None:
        if self._latest_decision is None:
            return
        for key, value in self._latest_decision.extras.items():
            if value is None:
                continue
            extras[f"icw/{key}"] = self._as_float(value)
        extras["icw/backend"] = 1.0 if self.current_backend == "neural" else 0.0
        extras["icw/abstained"] = 1.0 if self._latest_decision.abstain else 0.0
        extras["icw/dt_scale"] = self._as_float(self._latest_decision.dt_scale)

    def reset(
        self,
        backend: Optional[str] = None,
        states: Optional[torch.Tensor] = None,
    ) -> None:
        self._geometry.reset()
        if states is not None:
            self._geometry.observe(states)
        if backend is not None:
            self._intuition.reset(backend)
        else:
            self._intuition.reset()
        self._creativity.reset()
        self._latest_decision = None
        self._latest_snapshot = self._geometry.peek()


def attach_icw(neural_env, config: Optional[Dict] = None) -> ICWController:
    """Attach the ICW controller to ``neural_env`` and return it."""

    if config is None:
        with DEFAULT_CFG_PATH.open("r", encoding="utf-8") as cfg_file:
            config = yaml.safe_load(cfg_file) or {}
    else:
        config = dict(config)
    rung_dims = config.get("rungs", [128, 256, 512])
    geometry = LadderGeometryCache(rung_dims=rung_dims, device=torch.device(neural_env.torch_device))
    intuition_cfg = config.get("intuition", {})
    intuition = IntuitionController(
        hysteresis_lo=float(intuition_cfg.get("tau_lo", 0.08)),
        hysteresis_hi=float(intuition_cfg.get("tau_hi", 0.15)),
        abstain_quantile=float(intuition_cfg.get("abstain_quantile", 0.95)),
        residual_window=int(intuition_cfg.get("residual_window", 1024)),
        default_backend=config.get("default_backend", "ground-truth"),
    )
    creativity_cfg = config.get("creativity", {})
    creativity = CreativityPlanner(
        max_branches=int(creativity_cfg.get("max_branches", 2)),
        rupture_threshold=float(creativity_cfg.get("rupture_threshold", 0.2)),
    )
    wisdom_cfg = config.get("wisdom", {})
    wisdom = WisdomModule(
        disc_weight=float(wisdom_cfg.get("disc_weight", 1.0)),
        rupture_weight=float(wisdom_cfg.get("rupture_weight", 0.5)),
    )
    safety_cfg = config.get("safety", {})
    safety = SafetyAutomaton(
        torque_limit_scale=float(safety_cfg.get("torque_limit_scale", 1.1))
    )

    controller = ICWController(
        env=neural_env,
        geometry=geometry,
        intuition=intuition,
        creativity=creativity,
        wisdom=wisdom,
        safety=safety,
    )
    neural_env.register_auto_controller(controller)
    controller.reset(
        backend=config.get("default_backend", "ground-truth"),
        states=neural_env.states,
    )
    return controller

