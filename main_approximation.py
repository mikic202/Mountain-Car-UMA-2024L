import gymnasium as gym
from SARSA_approximation import SarsaWithAproximation
import numpy as np

print(gym.__version__)


def run_algorithm(env, algorithm: SarsaWithAproximation):
    observation, _ = env.reset(seed=123, options={})
    done = False
    sum_reward = 0
    while not done:
        action = np.array([algorithm.get_best_action(observation)])

        observation, r, terminated, truncated, info = env.step(action)
        sum_reward += r - 0.5

        done = terminated or truncated
    print(f"Sum of rewards: {sum_reward}")
    return terminated


def sarsa_single_epizode(
    enviornemt: gym.wrappers.time_limit.TimeLimit,
    solver: SarsaWithAproximation,
    t_max: int,
):
    t = 0
    state, _ = enviornemt.reset()
    terminated = False
    action = (
        np.array([solver.get_best_action(state)])
        if np.random.rand(1) > solver.epsilon
        else enviornemt.action_space.sample()
    )
    while t <= t_max and not terminated:
        observation, reward, terminated, truncated, info = enviornemt.step(action)
        next_action = (
            np.array([solver.get_best_action(observation)])
            if np.random.rand(1) > solver.epsilon
            else enviornemt.action_space.sample()
        )
        solver.update(state, action, reward, observation, next_action)
        t += 1
        action = next_action
        state = observation
        if terminated:
            print("Episode terminated")
            break


if __name__ == "__main__":

    env = gym.make("MountainCarContinuous-v0")

    ## Training

    # sarsa = SarsaWithAproximation(
    #     [
    #         lambda x, y: np.sin(x[0] * np.pi)
    #         - np.sin(x[1] * np.pi)
    #         + np.sin(y * np.pi),
    #         lambda x, y: np.cos(x[0] * np.pi)
    #         - np.cos(x[1] * np.pi)
    #         + np.cos(y * np.pi),
    #         lambda x, y: x[1] * x[0] * y,
    #         lambda x, y: np.tanh(x[1]) + np.tanh(x[0]) + np.tanh(y),
    #     ],
    #     epsilon=0.85,
    #     learning_rate=0.35,
    #     gamma=0.85,
    # )

    # for episode in range(250):
    #     sarsa_single_epizode(env, sarsa, 2000)
    #     print(f"Episode {episode} finished")

    # print(sarsa.q_table)

    # sarsa.save_to_file("test_q_table.npy")

    # print("Running algorithm")
    # env.close()
    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # run_algorithm(env, sarsa)

    # env.close()

    ## From saved file

    from_file_sarsa = SarsaWithAproximation(
        [
            lambda x, y: np.sin(x[0] * np.pi)
            - np.sin(x[1] * np.pi)
            + np.sin(y * np.pi),
            lambda x, y: np.cos(x[0] * np.pi)
            - np.cos(x[1] * np.pi)
            + np.cos(y * np.pi),
            lambda x, y: x[1] * x[0] * y,
        ]
    )
    from_file_sarsa.load_from_file("example.npy")
    print("Running algorithm frmo saved file")
    env.close()
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    run_algorithm(env, from_file_sarsa)

    env.close()
