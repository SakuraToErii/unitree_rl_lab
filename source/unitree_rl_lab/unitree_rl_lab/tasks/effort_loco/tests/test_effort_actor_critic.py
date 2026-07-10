from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


ACTOR_MODULE = Path(__file__).resolve().parents[1] / "agents" / "effort_actor_critic.py"


def _load_actor_module():
    spec = importlib.util.spec_from_file_location("effort_actor_critic_under_test", ACTOR_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {ACTOR_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EffortActorCriticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_actor_module()

    def _assert_small_gain_output(self, layer: torch.nn.Linear):
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))
        expected_norm = torch.full_like(layer.weight[:, 0], 0.01)
        torch.testing.assert_close(layer.weight.norm(dim=1), expected_norm, atol=1.0e-6, rtol=1.0e-5)

    def test_feed_forward_actor_is_residual_centered(self):
        torch.manual_seed(42)
        obs = {"policy": torch.zeros(8, 335), "critic": torch.zeros(8, 495)}
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        policy = self.module.EffortActorCritic(
            obs,
            obs_groups,
            num_actions=29,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            actor_output_gain=0.01,
        )

        self._assert_small_gain_output(policy.actor[-1])
        self.assertLess(policy.act_inference(obs).abs().max().item(), 0.01)

    def test_mha_actor_uses_the_same_output_gain(self):
        torch.manual_seed(42)
        obs = {"policy": torch.zeros(8, 335), "critic": torch.zeros(8, 495)}
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        policy = self.module.EffortActorCriticMHA(
            obs,
            obs_groups,
            num_actions=29,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            n_history=5,
            nheads=4,
            encoder_hidden_dim=256,
            is_learnable_pos_embedding=True,
            use_critic_mha=False,
            actor_term_dims=[3, 3, 3, 29, 29],
            critic_term_dims=[3, 3, 3, 3, 29, 29, 29],
            actor_output_gain=0.01,
        )

        self._assert_small_gain_output(policy.actor.trunk[-1])
        self.assertLess(policy.act_inference(obs).abs().max().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
