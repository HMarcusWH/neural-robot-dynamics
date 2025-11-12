from __future__ import annotations

import contextlib
import math
import subprocess
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


def _auto_rollout(
    *,
    num_envs: int = 8,
    horizon: int = 32,
    use_joint_act: bool = False,
    reset_step: int | None = None,
):
    patches = _patched_env()
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=num_envs,
            default_env_mode=BACKEND_ANALYTIC,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        controller = attach_icw(env, config={})
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.torch_device)
        joint_acts = torch.zeros((env.num_envs, env.joint_act_dim), device=env.torch_device)
        extras_seq = []
        for step in range(horizon):
            if reset_step is not None and step == reset_step:
                env.reset_envs(None)
            if use_joint_act:
                env.step_with_joint_act(joint_acts, env_mode=BACKEND_AUTO)
            else:
                env.step(actions, env_mode=BACKEND_AUTO)
            extras: dict[str, float] = {}
            env.get_extras(extras)
            extras_seq.append(extras)
        return extras_seq, controller


def _count_switches(values):
    switches = 0
    for left, right in zip(values, values[1:]):
        if left != right:
            switches += 1
    return switches


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
        expected = [(BACKEND_ANALYTIC, 0.0, 0.0)] * 3
        assert trace1 == expected
        assert trace2 == expected


def test_step_records_expected_metrics():
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
        env.step(actions, env_mode=BACKEND_AUTO)

        extras = {}
        env.get_extras(extras)
        assert extras["icw/backend"] == pytest.approx(0.0)
        assert extras["icw/abstained"] == pytest.approx(0.0)

        decision = controller._latest_decision
        assert decision.backend == BACKEND_ANALYTIC
        assert decision.extras["intuition/rupture"] == pytest.approx(0.0)
        assert decision.extras["creativity/triggered"] == pytest.approx(0.0)
        assert math.isinf(decision.extras["intuition/abstain_threshold"])
        for key in (
            "wisdom/certificates/C1_connectedness",
            "wisdom/certificates/C2_monotone_up",
            "wisdom/certificates/C3_compression_robust",
            "wisdom/certificates/C4_residual_agreement",
        ):
            assert decision.extras[key] == pytest.approx(1.0)
        assert decision.extras["wisdom/certificates"] == pytest.approx(1.0)
        assert decision.extras["wisdom/backend"] == pytest.approx(0.0)


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


@pytest.mark.parametrize("alias", ["analytic", "ground-truth"])
def test_neural_environment_accepts_backend_alias(alias):
    patches = _patched_env()
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode=alias,
            neural_integrator_cfg={},
            warp_env_cfg={},
        )
        assert env.default_env_mode == BACKEND_ANALYTIC
        env.set_env_mode(alias)
        assert env.env_mode == BACKEND_ANALYTIC
        env.set_env_mode(BACKEND_NEURAL)
        env.set_env_mode(alias)
        assert env.env_mode == BACKEND_ANALYTIC


@pytest.mark.gpu
def test_eval_passive_auto_requires_model():
    if not torch.cuda.is_available():
        pytest.skip("Passive auto eval requires CUDA to initialise the warp simulator")
    script = ROOT / "eval" / "eval_passive" / "eval_passive_motion.py"
    cmd = [
        sys.executable,
        str(script),
        "--env-name",
        "Cartpole",
        "--env-mode",
        "auto",
        "--num-rollouts",
        "2",
        "--rollout-horizon",
        "8",
        "--seed",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "requires a neural model" in combined


@pytest.mark.gpu
def test_auto_single_switch_and_extras():
    if not torch.cuda.is_available():
        pytest.skip("ICW telemetry sanity requires CUDA to drive the simulator")
    extras_seq, _ = _auto_rollout(num_envs=8, horizon=32)
    assert len(extras_seq) == 32
    backend_bits = [int(round(extras.get("icw/backend", 0.0))) for extras in extras_seq]
    assert all(bit in (0, 1) for bit in backend_bits)
    backend_labels = [extras.get("icw/backend_label") for extras in extras_seq]
    assert all(label in {BACKEND_ANALYTIC, BACKEND_NEURAL} for label in backend_labels)
    assert _count_switches(backend_bits) <= 10
    assert _count_switches(backend_labels) == _count_switches(backend_bits)
    for extras in extras_seq:
        assert "icw/intuition/rupture" in extras
        assert "icw/wisdom/disc" in extras
        assert "icw/backend" in extras
        assert "icw/backend_label" in extras
        assert "icw/abstained" in extras


@pytest.mark.gpu
def test_auto_reset_keeps_controller_clean():
    if not torch.cuda.is_available():
        pytest.skip("ICW telemetry reset check requires CUDA to run the simulator")
    extras_seq, controller = _auto_rollout(num_envs=8, horizon=16, reset_step=8)
    before = extras_seq[:8]
    after = extras_seq[8:]
    assert before and after
    assert "icw/intuition/rupture" in before[-1]
    assert "icw/intuition/rupture" in after[-1]
    assert after[-1]["icw/backend_label"] in {BACKEND_ANALYTIC, BACKEND_NEURAL}
    assert controller.current_backend in {BACKEND_ANALYTIC, BACKEND_NEURAL}


@pytest.mark.gpu
def test_step_with_joint_act_has_telemetry():
    if not torch.cuda.is_available():
        pytest.skip("Joint-act telemetry parity requires CUDA to execute the simulator")
    joint_extras, _ = _auto_rollout(num_envs=8, horizon=16, use_joint_act=True)
    action_extras, _ = _auto_rollout(num_envs=8, horizon=16, use_joint_act=False)
    assert len(joint_extras) == len(action_extras) == 16
    for joint_step, action_step in zip(joint_extras, action_extras):
        assert joint_step.keys() == action_step.keys()
        assert joint_step["icw/backend"] == pytest.approx(action_step["icw/backend"])
        assert joint_step["icw/backend_label"] == action_step["icw/backend_label"]
        assert joint_step["icw/abstained"] == pytest.approx(action_step["icw/abstained"])

