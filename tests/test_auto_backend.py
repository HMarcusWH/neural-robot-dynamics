import sys
import types
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Provide a lightweight cv2 stub so the environment wrapper can be imported in headless CI.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from envs import neural_environment as ne


class DummyIntegrator:
    joint_types = []

    def __init__(self, model, neural_model=None, **kwargs):
        pass

    def wrap2PI(self, states):
        return states

    def reset(self):
        pass

    def set_neural_model(self, neural_model):
        pass


class DummyEnv:
    def __init__(self):
        self.num_envs = 1
        self.dof_q_per_env = 2
        self.dof_qd_per_env = 2
        self.bodies_per_env = 1
        self.joint_act_dim = 2
        self.control_dim = 2
        self.control_limits = [(-1.0, 1.0)] * self.control_dim
        self.observation_dim = self.dof_q_per_env + self.dof_qd_per_env
        self.device = "cpu"
        self.robot_name = "dummy"
        self.abstract_contacts = types.SimpleNamespace(num_contacts_per_env=0)
        self.uses_generalized_coordinates = True
        self.model = object()
        self.sim_substeps = 1
        self.sim_dt = 0.02
        self.frame_dt = 0.02
        self.integrator = object()
        self.integrator_type = "ground-truth"
        self._control = types.SimpleNamespace(joint_act=torch.zeros(self.num_envs * self.joint_act_dim))
        self.state = types.SimpleNamespace(
            body_q=torch.zeros(self.num_envs * self.bodies_per_env, 7)
        )
        self.control_gains = [1.0] * self.control_dim
        self.controllable_dofs = list(range(self.control_dim))
        self.eval_collisions = False
        self._state_tensor = torch.zeros(self.num_envs, self.dof_q_per_env + self.dof_qd_per_env)

    @property
    def control(self):
        return self._control

    def assign_control(self, actions, control, state):
        control.joint_act = actions.clone().view(-1)

    def update(self):
        self._state_tensor = torch.ones_like(self._state_tensor)

    def reset(self):
        self._state_tensor.zero_()

    def reset_envs(self, env_ids=None):
        self.reset()

    def get_extras(self, extras):
        return extras

    def close(self):
        pass

    def compute_observations(self, *_, **__):
        pass

    def compute_cost_termination(self, *_, **__):
        pass

    def set_eval_collisions(self, flag):
        self.eval_collisions = flag

    def save_usd(self):
        pass

    def render(self):
        pass


class FixedBackendDecision:
    def __init__(self, backend):
        self.backend = backend
        self.backend_to_apply = backend
        self.action_delta = None
        self.dt_scale = 1.0
        self.abstain = False


class FixedBackendController:
    def __init__(self, backend: str):
        self._backend = backend
        self.after_calls = []

    @property
    def current_backend(self) -> str:
        return self._backend

    def before_step(self, actions):
        return FixedBackendDecision(self._backend)

    def after_step(self, states, backend: str):
        self.after_calls.append(backend)

    def populate_extras(self, extras):
        extras["icw/backend"] = 1.0 if self._backend == "neural" else 0.0

    def reset(self, backend=None, states=None):
        if backend is not None:
            self._backend = backend


def test_auto_backend_switch_and_logging():
    dummy_wp = types.SimpleNamespace(
        from_torch=lambda tensor: tensor.clone() if torch.is_tensor(tensor) else torch.tensor(tensor),
        to_torch=lambda arr: torch.as_tensor(arr),
        array=lambda data, **_: torch.as_tensor(data),
        device_to_torch=lambda device: torch.device(device) if not isinstance(device, torch.device) else device,
    )
    dummy_warp_utils = types.SimpleNamespace(
        eval_ik=lambda *_, **__: None,
        acquire_states_to_torch=lambda env, states: states.copy_(env._state_tensor),
        assign_states_from_torch=lambda env, states: setattr(env, "_state_tensor", states.clone()),
        eval_fk=lambda *_, **__: None,
    )

    with mock.patch.object(ne, "create_abstract_contact_env", side_effect=lambda *_, **__: DummyEnv()), \
        mock.patch.object(ne, "wp", dummy_wp), \
        mock.patch.object(ne, "warp_utils", dummy_warp_utils), \
        mock.patch.object(ne, "NeuralIntegrator", DummyIntegrator), \
        mock.patch.object(ne, "StatefulNeuralIntegrator", DummyIntegrator), \
        mock.patch.object(ne, "TransformerNeuralIntegrator", DummyIntegrator), \
        mock.patch.object(ne, "RNNNeuralIntegrator", DummyIntegrator), \
        mock.patch.object(ne, "IntegratorType", types.SimpleNamespace(NEURAL="neural")):

        env = ne.NeuralEnvironment(
            env_name="Dummy",
            num_envs=1,
            default_env_mode="ground-truth",
            neural_integrator_cfg={},
            warp_env_cfg={},
        )

        controller = FixedBackendController("neural")
        env.register_auto_controller(controller)

        actions = torch.full((env.num_envs, env.action_dim), 0.5, device=env.torch_device)
        env.step(actions, env_mode="auto")

        assert env._active_backend == "neural"
        assert controller.after_calls[-1] == "neural"

        extras = {}
        env.get_extras(extras)
        assert extras.get("icw/backend") == 1.0
