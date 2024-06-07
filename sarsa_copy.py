import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from discretization import get_discretized_array, find_closest, get_discretized_array_numpy

def run(episodes, position_length=30, velocity_length=30, forces_length=4, is_training=True, render=False):

    env = gym.make('MountainCarContinuous-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], position_length)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], velocity_length)    # Between -0.07 and 0.07
    available_forces = np.linspace(env.action_space.low[0], env.action_space.high[0], forces_length)
    available_forces = [np.array([elem], dtype=np.float32) for elem in available_forces]
    print(available_forces)
    if(is_training):
        q = np.zeros((len(pos_space), len(vel_space), len(available_forces))) # init a 20x20x3 array
    else:
        f = open(f'QTables/mountain_car_cont_f_{forces_length}_v_{velocity_length}_p_{position_length}.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 1.25/episodes # epsilon decay rate 0.0004
    rng = np.random.default_rng()   # random number generator
    reward_50_episodes = 0
    rewards_iteration_per_episode = np.empty((0, 2), float)  


    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal
        num_iteration = 0
        rewards=0
        while(not terminated and rewards>-2000):
            if is_training and rng.random() < epsilon:
                # Choose random action
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
            num_iteration += 1
        
        if terminated:
            reward_50_episodes += 1
        # print('--------------------------------')
        # print(a)
        # print(rewards)
        # print(terminated)
        # print('--------------------------------')
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_iteration_per_episode = np.vstack([rewards_iteration_per_episode, (rewards,num_iteration)])
        if i % 50 == 0:
            print(f"episode: {i}, rewards for 50 episodes: {reward_50_episodes}")
            reward_50_episodes = 0
    env.close()

    # Save Q table to file
    if is_training:
        f = open(f'QTables/mountain_car_cont_f_{forces_length}_v_{velocity_length}_p_{position_length}.pkl','wb')
        pickle.dump(q, f)
        f.close()

    # mean_rewards = np.zeros(episodes)
    # for t in range(episodes):
    #     mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    # plt.plot(mean_rewards)
    # plt.savefig(f'mountain_car_cont_4_v_30_p_30_act.png')
    return rewards_iteration_per_episode


def plot_img(forces_length, velocity_length, position_length):
    f = open(f'rewards/mountain_car_cont_f_{forces_length}_v_{velocity_length}_p_{position_length}_train.pkl', 'rb')
    average_rewards = pickle.load(f)
    f.close()
    # arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    plt.figure(1)
    plt.plot(average_rewards[:, 0], linestyle='-', color='b')
    plt.title('Average rewards in each episode during training')
    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')

    # Create the second figure
    plt.figure(2)
    plt.plot(average_rewards[:, 1], linestyle='--', color='r')
    plt.title('Average number of iteration in each episode')
    plt.xlabel('episodes')
    plt.ylabel('iterations')

    # Show the plots
    plt.show()
    pass

if __name__ == '__main__':
    position_length=28 
    velocity_length=28 
    forces_length=4
    train_episodes=2000
    run_episodes=2
    train_iterations_times = 5
    # rewards_iteration_per_episodes = run(train_episodes, position_length, velocity_length, forces_length, is_training=True, render=False)

    # for i in range(train_iterations_times-1):
    #     rewards_iteration_per_episode_i = run(train_episodes, position_length, velocity_length, forces_length, is_training=True, render=False)
    #     rewards_iteration_per_episodes += rewards_iteration_per_episode_i
    # rewards_iteration_per_episodes = rewards_iteration_per_episodes / train_iterations_times

    # f = open(f'rewards/mountain_car_cont_f_{forces_length}_v_{velocity_length}_p_{position_length}_train.pkl','wb')
    # pickle.dump(rewards_iteration_per_episodes, f)
    # f.close()
    # rewards_iteration_per_episode = run(run_episodes, position_length, velocity_length, forces_length, is_training=False, render=False)
    # f = open(f'rewards/mountain_car_cont_f_{forces_length}_v_{velocity_length}_p_{position_length}_not_train.pkl','wb')
    # pickle.dump(rewards_iteration_per_episode, f)
    # f.close()
    plot_img(forces_length, velocity_length, position_length)
