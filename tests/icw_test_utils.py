from __future__ import annotations

import sys
import types
from typing import Iterable, Tuple

import torch

# Provide a lightweight cv2 stub so the environment wrapper can be imported in headless CI.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))


class DummyIntegrator:
    joint_types = []

    def __init__(self, model, neural_model=None, **kwargs):
        self._model = model

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
        self.integrator_type = "analytic"
        self._control = types.SimpleNamespace(joint_act=torch.zeros(self.num_envs * self.joint_act_dim))
        self.state = types.SimpleNamespace(
            body_q=torch.zeros(self.num_envs * self.bodies_per_env, 7)
        )
        self.control_gains = [1.0] * self.control_dim
        self.controllable_dofs = list(range(self.control_dim))
        self.eval_collisions = False
        self._state_tensor = torch.zeros(self.num_envs, self.dof_q_per_env + self.dof_qd_per_env)
        self.assign_calls = 0

    @property
    def control(self):
        return self._control

    def assign_control(self, actions, control, state):
        self.assign_calls += 1
        control.joint_act = actions.clone().view(-1)
        return control

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


def dummy_env_patches(ne_module) -> Iterable[Tuple]:
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
    return (
        (ne_module, "create_abstract_contact_env", lambda *_, **__: DummyEnv()),
        (ne_module, "wp", dummy_wp),
        (ne_module, "warp_utils", dummy_warp_utils),
        (ne_module, "NeuralIntegrator", DummyIntegrator),
        (ne_module, "StatefulNeuralIntegrator", DummyIntegrator),
        (ne_module, "TransformerNeuralIntegrator", DummyIntegrator),
        (ne_module, "RNNNeuralIntegrator", DummyIntegrator),
    )
