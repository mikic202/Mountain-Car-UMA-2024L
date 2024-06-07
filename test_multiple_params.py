import numpy as np
from main_approximation import sarsa_single_epizode, run_algorithm
from SARSA_approximation import SarsaWithAproximation
import gymnasium as gym


ITERATIONS_PER_PARAMETERS_SET = 3

epsilons = np.arange(0.75, 0.9, 0.05)
gammas = np.arange(0.75, 0.9, 0.05)
lrs = np.arange(0.25, 0.5, 0.05)

first_set_of_functions = [
    lambda _, __: 1,
    lambda x, y: x[0] + x[1] + y,
    lambda x, y: x[1] ** 2 + x[0] ** 2 + y**2,
]

second_set_of_functions = [
    lambda _, __: 1,
    lambda x, y: x[0] + x[1] + y,
    lambda x, y: x[1] ** 2 + x[0] ** 2 + y**2,
    lambda x, y: x[1] ** 3 + x[0] ** 3 + y**3,
]

third_set_of_functions = [
    lambda _, __: 1,
    lambda x, y: x[0] + x[1] + y,
    lambda x, y: x[1] ** 2 + x[0] ** 2 + y**2,
    lambda x, y: x[1] ** 3 + x[0] ** 3 + y**3,
    lambda x, y: x[1] ** 4 + x[0] ** 4 + y**4,
]


fourth_set_of_functions = [
    lambda _, __: 1,
    lambda x, y: x[0] + x[1] + y,
    lambda x, y: x[1] ** 2 + x[0] ** 2 + y**2,
    lambda x, y: x[1] ** 3 + x[0] ** 3 + y**3,
    lambda x, y: x[1] * x[0] * y,
]

fifth_set_of_functions = [
    lambda x, y: np.sin(x[0] * np.pi) - np.sin(x[1] * np.pi) + np.sin(y * np.pi),
    lambda x, y: np.cos(x[0] * np.pi) - np.cos(x[1] * np.pi) + np.cos(y * np.pi),
    lambda x, y: x[1] * x[0] * y,
]

sixth_set_of_functions = [
    lambda x, y: np.sin(x[0] * np.pi) - np.sin(x[1] * np.pi) + np.sin(y * np.pi),
    lambda x, y: np.cos(x[0] * np.pi) - np.cos(x[1] * np.pi) + np.cos(y * np.pi),
    lambda x, y: x[1] * x[0] * y,
    lambda x, y: 1 / (1 + np.exp(-x[1])) * 1 / (1 + np.exp(-x[0])),
    lambda x, y: np.tanh(x[1]) + np.tanh(x[0]) + np.tanh(y),
]


funcyion_sets = [
    first_set_of_functions,
    second_set_of_functions,
    third_set_of_functions,
    fourth_set_of_functions,
    fifth_set_of_functions,
    sixth_set_of_functions,
]


results = []

for epsilon in epsilons:
    for gamma in gammas:
        for lr in lrs:
            for function_num in range(len(funcyion_sets)):
                for iteration in range(ITERATIONS_PER_PARAMETERS_SET):
                    env = gym.make("MountainCarContinuous-v0")
                    sarsa = SarsaWithAproximation(
                        funcyion_sets[function_num],
                        epsilon=epsilon,
                        learning_rate=lr,
                        gamma=gamma,
                    )
                    for episode in range(250):
                        sarsa_single_epizode(env, sarsa, 1500)
                    outcome = run_algorithm(env, sarsa)
                    sarsa.save_to_file(
                        f"results/{epsilon}_{gamma}_{lr}_{function_num}_{outcome}_{iteration}.npy"
                    )
                    results.append(
                        (epsilon, gamma, lr, function_num, outcome, iteration)
                    )
                print(f"Finished {epsilon} {gamma} {lr} {function_num}")

np.save("results.npy", results)
