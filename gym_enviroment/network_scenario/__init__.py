from gymnasium.envs.registration import register

register(
     id="Network-v0",
     entry_point="network_scenario.envs:NetworkEnv",
     max_episode_steps=300,
)

#env = gym.make('Network_scanario-v0')