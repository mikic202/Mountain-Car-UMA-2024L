import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from discretization import get_discretized_array, find_closest, get_discretized_array_numpy

def run(episodes, is_training=True, render=False):

    env = gym.make('MountainCarContinuous-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07
    available_forces = np.linspace(env.action_space.low[0], env.action_space.high[0], 6)
    available_forces = [np.array([elem], dtype=np.float32) for elem in available_forces]
    print(available_forces)
    if(is_training):
        q = np.zeros((len(pos_space), len(vel_space), len(available_forces))) # init a 20x20x3 array
    else:
        f = open('Mountain-Car-UMA-2024L/mountain_car_cont_2_v_20_p_20_act.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 1.25/episodes # epsilon decay rate 0.0004
    rng = np.random.default_rng()   # random number generator
    reward_50_episodes = 0
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal
        a = 0
        rewards=0
        while(not terminated and rewards>-2000):

            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
                action = find_closest(available_forces, action)
                action_index = available_forces.index(action)
            else:
                action_index = np.argmax(q[state_p, state_v, :])
                action = available_forces[action_index]

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action_index] = q[state_p, state_v, action_index] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action_index]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            reward -= 0.5
            rewards+=reward
            a += 1
        
        if terminated:
            reward_50_episodes += 1
        # print('--------------------------------')
        # print(a)
        # print(rewards)
        # print(terminated)
        # print('--------------------------------')
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards
        if i % 50 == 0:
            print(f"episode: {i}, rewards for 50 episodes: {reward_50_episodes}")
            reward_50_episodes = 0
    env.close()

    # Save Q table to file
    if is_training:
        f = open('Mountain-Car-UMA-2024L/mountain_car_cont_6_v_20_p_20_act.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car_cont_6_v_20_p_20_act.png')
    pass

if __name__ == '__main__':
    run(5000, is_training=True, render=False)

    run(10, is_training=False, render=True)