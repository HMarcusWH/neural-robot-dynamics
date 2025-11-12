from __future__ import annotations

import sys
import contextlib
import types
from pathlib import Path
from typing import List
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from robot_icw_mvp.constants import (
    BACKEND_ABSTAIN,
    BACKEND_ANALYTIC,
    BACKEND_AUTO,
    BACKEND_NEURAL,
)

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from envs import neural_environment as ne

from tests.icw_test_utils import dummy_env_patches


class ScriptedDecision:
    def __init__(self, backend: str, action_delta=None, dt_scale: float = 1.0):
        self.backend = backend
        self.action_delta = action_delta
        self.dt_scale = dt_scale
        self.abstain = backend == BACKEND_ABSTAIN

    @property
    def backend_to_apply(self) -> str:
        if self.backend == BACKEND_ABSTAIN:
            return BACKEND_ANALYTIC
        return self.backend


class ScriptedController:
    def __init__(self, decisions: List[ScriptedDecision]):
        self._decisions = decisions
        self._cursor = 0
        self.after_calls: List[str] = []

    @property
    def current_backend(self) -> str:
        if self._cursor == 0:
            return BACKEND_ANALYTIC
        return self._decisions[min(self._cursor - 1, len(self._decisions) - 1)].backend_to_apply

    def before_step(self, actions):
        decision = self._decisions[min(self._cursor, len(self._decisions) - 1)]
        self._cursor += 1
        return decision

    def after_step(self, states, backend: str):
        self.after_calls.append(backend)

    def populate_extras(self, extras):
        extras.setdefault("icw/backend", 1.0 if self.current_backend == BACKEND_NEURAL else 0.0)
        extras.setdefault("icw/backend_label", self.current_backend)

    def reset(self, backend=None, states=None):
        if backend is not None:
            for decision in self._decisions:
                decision.backend = backend
        self._cursor = 0
        self.after_calls.clear()


def _with_dummy_env():
    patches = [
        mock.patch.object(target, attribute, value)
        for target, attribute, value in dummy_env_patches(ne)
    ]
    patches.append(mock.patch.object(ne, "IntegratorType", types.SimpleNamespace(NEURAL=BACKEND_NEURAL)))
    return patches


def test_auto_mode_switches_once_per_step():
    patches = _with_dummy_env()
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode=BACKEND_ANALYTIC,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        controller = ScriptedController([ScriptedDecision(BACKEND_NEURAL)])
        env.register_auto_controller(controller)
        actions = torch.full((env.num_envs, env.action_dim), 0.5, device=env.torch_device)
        with mock.patch.object(env, "set_env_mode", wraps=env.set_env_mode) as set_mode:
            env.step(actions, env_mode=BACKEND_AUTO)
            assert set_mode.call_count == 1
        assert controller.after_calls == [BACKEND_NEURAL]
        assert env._active_backend == BACKEND_NEURAL
        extras = {}
        env.get_extras(extras)
        assert extras["icw/backend_label"] == BACKEND_NEURAL


def test_abstain_path_reduces_dt_and_clamps_actions():
    patches = _with_dummy_env()
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode=BACKEND_ANALYTIC,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        # Attach full ICW controller with default config for safety logic.
        from robot_icw_mvp.adapters.nerd import attach_icw

        controller = attach_icw(env, config={})
        # Force the controller into neural mode first so that an abstain transition is observable.
        controller._intuition.reset(BACKEND_NEURAL)
        high_actions = torch.full((env.num_envs, env.action_dim), 5.0, device=env.torch_device)
        env.step(high_actions, env_mode=BACKEND_AUTO)
        # SafetyAutomaton should clamp to scaled limits.
        applied = env.joint_acts.view(env.num_envs, env.joint_act_dim)
        assert torch.all(applied <= 1.1 + 1e-6)
        assert env.env.sim_substeps > env.sim_substeps_gt
        extras = {}
        env.get_extras(extras)
        assert extras["icw/backend"] == 0.0  # analytic fallback
        assert extras["icw/abstained"] == 1.0

