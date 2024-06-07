import gymnasium as gym
from SARSA_approximation import SarsaWithAproximation
import numpy as np

print(gym.__version__)


def run_algorithm(env, algorithm):
    observation, _ = env.reset(seed=123, options={})
    done = False
    while not done:
        action = np.array([algorithm.get_best_action(observation)])
        # print(action)
        observation, _, terminated, truncated, info = env.step(action)

        done = terminated or truncated
    return terminated


def sarsa_single_epizode(
    enviornemt: gym.wrappers.time_limit.TimeLimit,
    solver: SarsaWithAproximation,
    t_max: int,
):
    t = 0
    state, _ = enviornemt.reset()
    terminated = False
    while t <= t_max and not terminated:
        action = (
            np.array([solver.get_best_action(state)])
            if np.random.rand(1) > solver.epsilon
            else enviornemt.action_space.sample()
        )
        observation, reward, terminated, truncated, info = enviornemt.step(action)
        next_action = (
            np.array([solver.get_best_action(observation)])
            if np.random.rand(1) > solver.epsilon
            else enviornemt.action_space.sample()
        )
        solver.update(state, action, reward, observation, next_action)
        state = observation
        t += 1
        if terminated:
            print("Episode terminated")
            break


if __name__ == "__main__":

    env = gym.make("MountainCarContinuous-v0")

    # works
    # lambda x, y: np.sin(x[0] * np.pi) - np.sin(x[1] * np.pi) + np.sin(y * np.pi),
    # lambda x, y: np.cos(x[0] * np.pi) - np.cos(x[1] * np.pi) + np.cos(y * np.pi),
    # lambda x, y: x[1] * x[0] * y,

    # even better
    # lambda x, y: np.sin(x[0] * np.pi) - np.sin(x[1] * np.pi) + np.sin(y * np.pi),
    # lambda x, y: np.cos(x[0] * np.pi) - np.cos(x[1] * np.pi) + np.cos(y * np.pi),
    # lambda x, y: x[1] * x[0] * y,
    # lambda x, y: np.exp(-x[1]) * np.exp(x[0]),

    # udało się dojechać do końca

    # lambda x, y: np.sin(x[0] * np.pi) - np.sin(x[1] * np.pi) + np.sin(y * np.pi),
    # lambda x, y: np.cos(x[0] * np.pi) - np.cos(x[1] * np.pi) + np.cos(y * np.pi),
    # lambda x, y: x[1] * x[0] * y,
    # lambda x, y: 1 / (1 + np.exp(-x[1])) * 1 / (1 + np.exp(-x[0])),
    # lambda x, y: np.tanh(x[1]) + np.tanh(x[0]) + np.tanh(y),

    # even better finish and quick_finish
    # lambda x, y: np.sin(x[0] * np.pi) - np.sin(x[1] * np.pi) + np.sin(y * np.pi),
    # lambda x, y: np.cos(x[0] * np.pi) - np.cos(x[1] * np.pi) + np.cos(y * np.pi),
    # lambda x, y: x[1] * x[0] * y,
    # lambda x, y: np.tanh(x[1]) + np.tanh(x[0]) + np.tanh(y),

    sarsa = SarsaWithAproximation(
        [
            lambda x, y: np.sin(x[0] * np.pi)
            - np.sin(x[1] * np.pi)
            + np.sin(y * np.pi),
            lambda x, y: np.cos(x[0] * np.pi)
            - np.cos(x[1] * np.pi)
            + np.cos(y * np.pi),
            lambda x, y: x[1] * x[0] * y,
            # lambda x, y: 1 / (1 + np.exp(-x[1])) * 1 / (1 + np.exp(-x[0])),
            lambda x, y: np.tanh(x[1]) + np.tanh(x[0]) + np.tanh(y),
        ],
        epsilon=0.85,
        learning_rate=0.4,
        gamma=0.85,
    )

    for episode in range(200):
        sarsa_single_epizode(env, sarsa, 1500)
        print(f"Episode {episode} finished")

    sarsa.epsilon = 0.7

    for episode in range(50):
        sarsa_single_epizode(env, sarsa, 1500)
        print(f"Episode {200+episode} finished")

    print(sarsa.q_table)

    sarsa.save_to_file("test2.npy")
    # sarsa.load_from_file("test_finished.npy")

    print("Running algorithm")
    env.close()
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    run_algorithm(env, sarsa)

    env.close()
