from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper
from gymnasium.wrappers import NormalizeObservation
from rl_zoo3.wrappers import ObservationNormalizationWrapper
from gymnasium.wrappers import NormalizeReward
from gymnasium.wrappers import FlattenObservation

try:
    import pybullet_envs_gymnasium
except ImportError:
    pass

import numpy as np


def create_normalized_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = ObservationNormalizationWrapper(env)
        #env = FlattenObservation(env)
        #env = NormalizeReward(env)
        return env
    return make_env


try:
    from env.antennaEnv_V2 import AntennaPlacementEnv
except ImportError:
    AntennaPlacementEnv = None
    print("Custom Antenna Environment failed to import")

register(
    # unique identifier for the env `name-version`
    id="antenna3x4-v2.0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=AntennaPlacementEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
    )

del AntennaPlacementEnv
try:
    from env.antennaEnv_V3 import AntennaPlacementEnvSeq
except ImportError:
    AntennaPlacementEnv = None
    print("Custom Antenna Environment failed to import")

#### antenna3x4-v3
register(
    id="antenna3x4-v3",
    entry_point=AntennaPlacementEnvSeq,
    max_episode_steps=25,
)

#### end
    

