import pickle
import numpy as np


def get_report(average_rewards):
    rewards = average_rewards[:, 0]
    iterations = average_rewards[:, 1]
    terminated = average_rewards[:, 2]
    print(f"avg iterations: {np.mean(iterations)}\navg reward: {np.mean(rewards)}\navg terminated: {np.mean(terminated)}")

available_parameters = [
    (2,5,5),
    (2,10,10),
    (3,3,3), 
    (10,5,5),
    (10,10,10),
    (10,30,30),
    (100,30,30),
    (100,300,300)
]

isSarsa = False
for param in available_parameters:
    force_count, velocity_count, position_count = param
    if isSarsa:
        f = open(f'SARSA/rewards/mountain_car_cont_f_{force_count}_v_{velocity_count}_p_{position_count}_train.pkl', 'rb')
    else:
        f = open(f'QLearning/rewards/mountain_car_cont_f_{force_count}_v_{velocity_count}_p_{position_count}_train.pkl', 'rb')
    average_rewards = pickle.load(f)
    print(f"{force_count}\t{velocity_count}\t{position_count}")
    get_report(average_rewards)
    print('----------------------------------------------------------------')
    f.close()