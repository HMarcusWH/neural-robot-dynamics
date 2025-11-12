from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from unittest import mock

import torch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from robot_icw_mvp.adapters.nerd import attach_icw
from robot_icw_mvp.constants import (
    BACKEND_ANALYTIC,
    BACKEND_AUTO,
    BACKEND_NEURAL,
)
from robot_icw_mvp.creativity.planner import CreativityProposal
from robot_icw_mvp.wisdom.memory_lenses import WisdomSummary

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from envs import neural_environment as ne

from tests.icw_test_utils import dummy_env_patches


def _patched_env():
    patches = [
        mock.patch.object(target, attribute, value)
        for target, attribute, value in dummy_env_patches(ne)
    ]
    patches.append(mock.patch.object(ne, "IntegratorType", types.SimpleNamespace(NEURAL=BACKEND_NEURAL)))
    return patches


def test_certificates_block_backend_and_creativity():
    patches = _patched_env()
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode=BACKEND_ANALYTIC,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        controller = attach_icw(env, config={})
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.torch_device)

        certificate_keys = [
            "C1_connectedness",
            "C2_monotone_up",
            "C3_compression_robust",
            "C4_residual_agreement",
        ]

        for key in certificate_keys:
            summary = WisdomSummary(metrics={"disc": 0.0}, certificates={k: True for k in certificate_keys}, certificates_passed=False)
            summary.certificates[key] = False

            def _precheck(_snapshot, *_, summary=summary, **__):
                controller._wisdom._latest_summary = summary
                return summary

            with mock.patch.object(controller._wisdom, "precheck", side_effect=_precheck), \
                mock.patch.object(controller._wisdom, "summarize", side_effect=_precheck), \
                mock.patch.object(
                    controller._creativity,
                    "propose",
                    return_value=CreativityProposal(delta_actions=torch.ones_like(actions), metadata={}, triggered=True),
                ):
                env.step(actions, env_mode=BACKEND_AUTO)
                applied = env.joint_acts.view(env.num_envs, env.joint_act_dim)
                assert torch.allclose(applied, torch.zeros_like(applied))
                assert env._active_backend == BACKEND_ANALYTIC
                decision = controller._latest_decision
                assert decision.backend == "abstain"
                assert decision.extras[f"wisdom/certificates/{key}"] == 0.0


def test_reset_preserves_runtime_backend():
    patches = _patched_env()
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode=BACKEND_ANALYTIC,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        controller = attach_icw(env, config={})
        controller._intuition.reset(BACKEND_NEURAL)
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.torch_device)
        env.step(actions, env_mode=BACKEND_AUTO)
        current = controller.current_backend
        controller.reset(backend=None, states=env.states)
        assert controller.current_backend == current
        controller.reset(backend=BACKEND_ANALYTIC, states=env.states)
        assert controller.current_backend == BACKEND_ANALYTIC


def test_deterministic_trace_with_seed():
    patches = _patched_env()
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode=BACKEND_ANALYTIC,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        controller = attach_icw(env, config={})
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.torch_device)

        def run_trace():
            trace = []
            for _ in range(3):
                env.step(actions, env_mode=BACKEND_AUTO)
                extras = {}
                env.get_extras(extras)
                trace.append((env._active_backend, extras["icw/backend"], extras["icw/abstained"]))
            return trace

        trace1 = run_trace()
        env.reset()
        controller.reset(backend=BACKEND_ANALYTIC, states=env.states)
        trace2 = run_trace()
        assert trace1 == trace2


def test_attach_requires_yaml_when_config_missing():
    patches = _patched_env()
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode=BACKEND_ANALYTIC,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        with mock.patch.dict(sys.modules, {"yaml": None}):
            with pytest.raises(RuntimeError):
                attach_icw(env, config=None)
