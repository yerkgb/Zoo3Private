from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper
from gymnasium.wrappers import NormalizeObservation

try:
    import pybullet_envs_gymnasium
except ImportError:
    pass

import numpy as np

class ObservationNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mean = np.zeros(env.observation_space['antenna_positions'].shape, dtype=np.float32)
        self.std = np.ones(env.observation_space.shape['antenna_positions'], dtype=np.float32)
        
    def observation(self, observation):
        normalized_obs = (observation - self.mean) / (self.std + 1e-8)
        return normalized_obs

def create_normalized_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = ObservationNormalizationWrapper(env)
        return env
    return make_env


try:
    from env.antennaEnv import AntennaPlacementEnv
except ImportError:
    AntennaPlacementEnv = None
    print("Custom Antenna Environment failed to import")

register(
    # unique identifier for the env `name-version`
    id="antenna4x4-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=AntennaPlacementEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=25,
    )
del AntennaPlacementEnv

try:
    from env.antennaEnv_V2 import AntennaPlacementEnv
except ImportError:
    AntennaPlacementEnv = None
    print("Custom Antenna Environment failed to import")

register(
    # unique identifier for the env `name-version`
    id="antenna4x4-v2",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=AntennaPlacementEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
    )

del AntennaPlacementEnv
try:
    from env.antennaEnv_V1_1 import AntennaPlacementEnv
except ImportError:
    AntennaPlacementEnv = None
    print("Custom Antenna Environment failed to import")


register(
    id="antenna4x4-v1_1",
    entry_point=AntennaPlacementEnv,
    max_episode_steps=25,
)

register(
    id="antenna3x4-v1_1",
    entry_point=AntennaPlacementEnv,
    max_episode_steps=25,
)

register(
    id="antenna4x4-v1.1_n",
    entry_point=create_normalized_env("antenna4x4-v1.1"),
    max_episode_steps=25,
)






# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env



for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )
    

