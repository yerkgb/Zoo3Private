from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper
from gymnasium.wrappers import NormalizeObservation
from rl_zoo3.wrappers import ObservationNormalizationWrapper
from gymnasium.wrappers import NormalizeReward
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import VecNormalize

# try:
#     import pybullet_envs_gymnasium
# except ImportError:
#     pass
from stable_baselines3.common.vec_env import VecNormalize

# try:
#     import pybullet_envs_gymnasium
# except ImportError:
#     pass

import numpy as np


def create_normalized_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = ObservationNormalizationWrapper(env)
        #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        #env = FlattenObservation(env)
        #env = NormalizeReward(env)
        return env
    return make_env




try:
    from env.antennaEnv_V1_2 import AntennaPlacementEnv
except ImportError:
    AntennaPlacementEnv = None
    print("Custom Antenna Environment failed to import")

#### antenna3x4-v1.2
register(
    id="antenna-v1.2",
    entry_point=AntennaPlacementEnv,
    max_episode_steps=25,
)



#### end
    

