"""An example of customizing PPO to leverage a centralized critic.

Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.

Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.

See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

import argparse
import numpy as np
from gym.spaces import Discrete
import os
import gfootball.env as football_env
import gym
import numpy as np
import ray
import torch
from gfootball import env as fe
from gym import wrappers
import ray
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from rldm.utils import football_tools as ft
from rldm.utils import gif_tools as gt
from rldm.utils import system_tools as st
from rldm.scripts.TorchCentralizedCriticModel import TorchCentralizedCriticModel
from ray.tune.registry import register_env

import torch
import torch.nn as nn

TEAMMATE_ACTION = "teammate_action"


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        [(_, teammate_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[TEAMMATE_ACTION] = teammate_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(sample_batch[SampleBatch.CUR_OBS], policy.device),
            convert_to_torch_tensor(sample_batch[SampleBatch.ACTIONS], policy.device),
            convert_to_torch_tensor(sample_batch[TEAMMATE_ACTION], policy.device)) \
            .cpu().detach().numpy()
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[TEAMMATE_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[SampleBatch.ACTIONS],
        train_batch[TEAMMATE_ACTION])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out),
    }

CCPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="CCPPOTorchPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_init=setup_torch_mixins,
    mixins=[
        TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class(config):
    if config["framework"] == "torch":
        return CCPPOTorchPolicy


CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer",
    default_policy=CCPPOTorchPolicy,
    get_policy_class=get_policy_class,
)

if __name__ == "__main__":
    # args = parser.parse_args()
    debug = False
    n_timesteps = 100000000
    ray.init(num_cpus=8, num_gpus=1, local_mode=debug)
    env_name = ft.n_players_to_env_name(3, True) # hard-coding auto GK
    register_env(env_name, lambda _: ft.RllibGFootball(env_name=env_name))
    ModelCatalog.register_custom_model("cc_model", TorchCentralizedCriticModel)
    obs_space, act_space = ft.get_obs_act_space(env_name)
    policies = {
        'player_0' : (None, obs_space['player_0'], act_space['player_0'], {}),
        'player_1' : (None, obs_space['player_1'], act_space['player_1'], {}),
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id
    
    config = {
        "env": env_name,
        "framework": "torch",
        "batch_mode": "complete_episodes",
        'lr': 0.00022602718266055705,
        'gamma': 0.9936809332376452,
        'lambda': 0.9517171675473532,
        'kl_target': 0.010117093480119358,
        'kl_coeff': 1.0,
        'clip_param': 0.20425701146213993,
        'vf_loss_coeff': 0.3503035138680095,
        'vf_clip_param': 1.4862186106326711,
        'entropy_coeff': 0.0004158966184268587,
        'num_sgd_iter': 16,
        'train_batch_size': 2_800,
        'rollout_fragment_length': 100,
        'sgd_minibatch_size': 128,
        'num_workers': 7, # one goes to the trainer
        'num_envs_per_worker': 1,
        'num_gpus': 1,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'log_level': 'INFO' if not debug else 'DEBUG',
        'ignore_worker_failures': False,
        'horizon': 500,
        'model': {
            "custom_model": "cc_model",
            # 'vf_share_layers': "true",
            # 'use_lstm': "true",
            # 'max_seq_len': 13,
            # 'fcnet_hiddens': [256, 256],
            # 'fcnet_activation': "tanh",
            # 'lstm_cell_size': 256,
            # 'lstm_use_prev_action': "true",
            # 'lstm_use_prev_reward': "true",
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        }
    }
    config['callbacks'] = ft.FootballCallbacks
    filename_stem = os.path.basename(__file__).split(".")[0]
    policy_type = 'independent'
    experiment_name =f"{filename_stem}_{env_name}_{policy_type}_{n_timesteps}"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_dir = os.path.join(script_dir, '..', '..', 'logs')
    stop = {
        "timesteps_total": n_timesteps,
    }

    results = tune.run(CCTrainer, 
        config=config, 
        reuse_actors=False,
        raise_on_failed_trial=True,
        fail_fast=True,
        max_failures=0,
        num_samples=1,
        stop=stop,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir=local_dir,
        verbose=1 if not debug else 3)