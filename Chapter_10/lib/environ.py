import typing as tt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class StocksEnv(gym.Env):
    spec = EnvSpec("StocksEnv-v0")

    