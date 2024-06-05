import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from discretization import get_discretized_array, find_closest, get_discretized_array_numpy
from collections import defaultdict
import matplotlib.pyplot as plt

def train(episodes, 
        learning_rate = 0.9,
        discount_coeff = 0.9,  # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
        alpha=None,
        render=False, 
        velocity_step=0.008,
        position_step=0.08,
        force_step=1,
        precision=3
        ):
    env = gym.make("MountainCarContinuous-v0", render_mode='human' if render else None)

    # good_q = None
    max_episode_steps = 1500
    max_episode_steps = 2000
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset()

    available_positions = get_discretized_array(env.observation_space.low[0], env.observation_space.high[0], position_step, precision)
    available_velocities = get_discretized_array(env.observation_space.low[1], env.observation_space.high[1], velocity_step, precision)
    available_forces = get_discretized_array_numpy(env.action_space.low[0], env.action_space.high[0], force_step, precision)
    # del available_forces[0]
    # del available_forces[0]
    # del available_forces[0]
    # del available_forces[0]
    num_positions = len(available_positions)
    num_velocities = len(available_velocities)
    num_forces = len(available_forces)
    q = np.zeros((num_positions, num_velocities, num_forces))

    epsilon = 1
    epsilon_diff = 0.0002
    rewards_per_episodes = np.zeros(episodes)
    for i in range(episodes):
        a = 0
        state_position, state_velocity = env.reset()[0]  
        state_position = find_closest(available_positions, state_position)
        state_velocity = find_closest(available_velocities, state_velocity)
        state_position_index = available_positions.index(state_position)
        state_velocity_index = available_velocities.index(state_velocity)
        terminated = False 
        truncated = False   
        pair_counts_pos = defaultdict(int)
        pair_counts_vel = defaultdict(int)

        for _ in range(max_episode_steps):
            r = np.random.rand()
            if r < epsilon:
                action_force = find_closest(available_forces, env.action_space.sample())
                action_force_index = available_forces.index(action_force)
            else:
                action_force_index = np.argmax(q[state_position_index, state_velocity_index])
                action_force = available_forces[action_force_index]

# 
            pair = (state_position_index)
            pair_counts_pos[pair] += 1
            pair = (state_velocity_index)
            pair_counts_vel[pair] += 1
#   
            new_state,reward,terminated,truncated,_ = env.step(action_force)       


            new_state_position = find_closest(available_positions, new_state[0])
            new_state_velocity = find_closest(available_velocities, new_state[1])
            new_state_position_index = available_positions.index(new_state_position)
            new_state_velocity_index = available_velocities.index(new_state_velocity)
            
            q[state_position_index, state_velocity_index, action_force_index] = q[state_position_index, state_velocity_index, action_force_index] + learning_rate * (
                reward + discount_coeff * np.max(q[new_state_position_index, new_state_velocity_index]) - q[state_position_index, state_velocity_index, action_force_index]
            ) 
            if terminated:
                print('--------------------------------------------------------')
                print('terminated!')
                print('--------------------------------------------------------')
                # maybe_good_q = run(q,velocity_step, position_step, force_step, precision)
                # if maybe_good_q is not None:
                #     print("good q found!!!")
                #     good_q = maybe_good_q
                #     break


            state_position, state_velocity = new_state
            state_position = find_closest(available_positions, state_position)
            state_velocity = find_closest(available_velocities, state_velocity)
            state_position_index = available_positions.index(state_position)
            state_velocity_index = available_velocities.index(state_velocity)
            if terminated:
                break
            a = a + 1

        epsilon = max(epsilon - epsilon_diff, 0)
        if i % 50 == 0:
            print(a)
            print(f"episode: {i}, eps: {epsilon}")
        if(epsilon==0):
            learning_rate = 0.0001

        if reward >= 1:
            print('reward!!!!!!')
            rewards_per_episodes[i] = 1
            # if good_q is not None:
            #     break
        



    env.close()
    return (q, rewards_per_episodes, pair_counts_pos, pair_counts_vel, good_q)


def run(q, 
        velocity_step=0.03,
        position_step=0.2,
        force_step=0.1,
        precision=3,
        render=True
        ): 
    env = gym.make("MountainCarContinuous-v0", render_mode='human' if render else None)
    available_positions = get_discretized_array(env.observation_space.low[0], env.observation_space.high[0], position_step, precision)
    available_velocities = get_discretized_array(env.observation_space.low[1], env.observation_space.high[1], velocity_step, precision)
    available_forces = get_discretized_array_numpy(env.action_space.low[0], env.action_space.high[0], force_step, precision)
    state_position, state_velocity = env.reset()[0]  
    state_position = find_closest(available_positions, state_position)
    state_velocity = find_closest(available_velocities, state_velocity)
    state_position_index = available_positions.index(state_position)
    state_velocity_index = available_velocities.index(state_velocity)
    terminated = False 
    for i in range(2000):
        action_force_index = np.argmax(q[state_position_index, state_velocity_index])
        action_force = available_forces[action_force_index]
        new_state,reward,terminated,truncated,_ = env.step(action_force)
        # new_state_position = find_closest(available_positions, new_state[0])
        # new_state_velocity = find_closest(available_velocities, new_state[1])
        # new_state_position_index = available_positions.index(new_state_position)
        # new_state_velocity_index = available_velocities.index(new_state_velocity)

        state_position, state_velocity = new_state
        state_position = find_closest(available_positions, state_position)
        state_velocity = find_closest(available_velocities, state_velocity)
        state_position_index = available_positions.index(state_position)
        state_velocity_index = available_velocities.index(state_velocity)
        if terminated:
            print('good strategy!')
            return q
    return None

            


q, r, pcp, pcv, goodq= train(4000)

# np.save('q4.npy', goodq)


def plot_pair_counts(pair_counts):
    # Extract pairs and their counts
    sorted_pair_counts = dict(sorted(pair_counts.items()))

    pairs = list(pair_counts.keys())
    counts = list(pair_counts.values())
    
    # Convert pairs to strings for better readability on the plot
    pair_strings = [f"{pair}" for pair in pairs]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(pair_strings, counts, color='skyblue')
    plt.xlabel('Pairs')
    plt.ylabel('Frequency')
    plt.title('Frequency of Pairs')
    plt.xticks(rotation=90)  # Rotate x labels for better readability
    plt.tight_layout()


# Example usage
# plot_pair_counts(pcp)
# plot_pair_counts(pcv)
# plt.show()
# pass
# # np.save('q2.npy', q)

# q = np.load('q4.npy')
# run(q)


# pass

# import gymnasium as gym
# import numpy as np

# # Create the MountainCarContinuous environment
# env = gym.make("MountainCarContinuous-v0")


# # Reset the environment to get the initial observation
# observation, info = env.reset(seed=42)

# # Parameters for the loop
# num_steps = 1500  # Number of steps to run in the environment
# done = False

# # Run a loop to interact with the environment
# for step in range(num_steps):
#     if done:
#         break
    
#     # Render the environment (optional)
#     env.render()

#     # Sample a random action from the action space
#     action = env.action_space.sample()

#     # Take a step in the environment with the sampled action
#     observation, reward, done, truncated, info = env.step(action)

#     # Print out some details for debugging
#     print(f"Step: {step}, Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}")

# # Close the environment when done
# env.close()
