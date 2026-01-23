import gymnasium as gym
import ptan
from ptan.experience import ExperienceFirstLast

import typing as tt
from ray import tune

import torch
from torch import nn
from torch import optim
import numpy as np

from ignite.engine import Engine

from lib import dqn_model, common, dqn_extra

NAME = "05_prio_replay"
PRIO_REPLAY_ALPHA = 0.6


BEST_PONG = common.Hyperparams(
    env_name="PongNoFrameskip-v4",
    stop_reward=18.0,
    run_name="pong",
    replay_size=100_000,
    replay_initial=10_000,
    target_net_sync=1000,
    epsilon_frames=100_000,
    epsilon_final=0.02,
    learning_rate=8.839010139505506e-05,
    gamma=0.99,
    episodes_to_solve=333,
)

