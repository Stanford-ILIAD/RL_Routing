import gym
from gym.envs.registration import register


env_name = 'RoadNetwork-v0'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]
register(
    id=env_name,
    entry_point='gym_roadnetwork.envs:' + env_name[:-3],
)