import gymnasium as gym
from SARSA_approximation import SarsaWithAproximation
import numpy as np

print(gym.__version__)


def run_algorithm(env, algorithm):
    observation, _ = env.reset(seed=123, options={})
    done = False
    while not done:
        action = np.array([algorithm.get_best_action(observation)])
        observation, _, terminated, truncated, info = env.step(action)

        done = terminated or truncated


env = gym.make("MountainCarContinuous-v0", render_mode="human")

sarsa = SarsaWithAproximation(
    [lambda x, y: x[0] + x[1] + y, lambda x, y: x[1] * x[0] * y]
)

run_algorithm(env, sarsa)

env.close()
