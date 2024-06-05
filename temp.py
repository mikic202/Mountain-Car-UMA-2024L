import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import csv

def train(episodes, 
        render=False, 
        learning_rate = 0.9,
        discount_coeff = 0.5,  # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
        alpha=None
        ):

    env = gym.make('CliffWalking-v0', render_mode='human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 48 x 4 array

    epsilon = 1         # 1 = 100% random actions
    epsilon_diff = 0.001        # epsilon decay rate. 1/0.001 = 10,00

    rewards_per_episodes = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 47, 0=top left corner, 47=bottom right corner
        terminated = False # True when fall in cliff or reached goal
        truncated = False       
        while(not terminated and not truncated):
            if np.random.rand() < epsilon:
                action = env.action_space.sample() # actions: 0=up,1=right,2=down,3=left
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)
            if alpha is not None:
                learning_rate = 1/((i+1)**alpha) 
            if new_state == 47:
                reward = 1
                terminated = True
            if reward == -100:
                 terminated = True
            q[state,action] = q[state,action] + learning_rate * (
				reward + discount_coeff * np.max(q[new_state,:]) - q[state,action]
			)
            state = new_state

				
        epsilon = max(epsilon - epsilon_diff, 0)

        if(epsilon==0):
            learning_rate = 0.0001

        if reward == 1:
            rewards_per_episodes[i] = 1


    env.close()
    return (q, rewards_per_episodes)


def run(episodes, q, render=True ):
	env = gym.make('CliffWalking-v0', render_mode='human' if render else None)
	# print('---------------')
	# for i in range(48):
	# 	strrr = str(i) + ": "
	# 	for j in range(4):
	# 		strrr += f"{q[i,j]}, "
	# 	print(strrr)
	# print('---------------')
	for i in range(episodes):
		state = env.reset()[0]
		terminated = False      # True when fall in hole or reached goal
		truncated = False       
		while(not terminated and not truncated):
			action = np.argmax(q[state,:])
			new_state,reward,terminated,truncated,_ = env.step(action)
            
			if new_state == 47:
				reward = 1
			state = new_state
            


	env.close()


def import_q_from_file(file_name):
    q =  np.zeros((48, 4)) 
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        i = 0
        for row in csv_reader:
            q[i] = [row[1], row[2], row[3], row[4]]
            i += 1
    return q


if __name__ == "__main__":

    # q = import_q_from_file('ep_d_r_0_001_lr_alpha_1_dc_0_9.csv')
    # q, rewards_per_episodes  = train(1500, render=False , learning_rate=0.09, discount_coeff=0.5)
    # run(2, q)
    ###################################################
    total_rewards = 0
    every_reward_per_episodes = []
    nm_episodes = 1500
    for i in range(2):
        q, rewards_per_episodes  = train(nm_episodes, render=False, learning_rate=0.9, discount_coeff=0.5)
        # print('---------------')
        # for i in range(48):
        #     strrr = str(i) + ": "
        #     for j in range(4):
        #         strrr += f"{q[i,j]}, "
        #     print(strrr)
        # print('---------------')
        run(1, q, True)
        every_reward_per_episodes.append(rewards_per_episodes)
        total_rewards += sum(rewards_per_episodes)
    avg_reward = total_rewards / 10

    total_rewards_per_episode_arr = []
    print(len(every_reward_per_episodes))
    print(len(rewards_per_episodes))
    for i in range(len(rewards_per_episodes)):
        total_rewards_per_episode = 0
        for j in range(len(every_reward_per_episodes)):
            if every_reward_per_episodes[j][i] == 1:
                total_rewards_per_episode += 1
        total_rewards_per_episode_arr.append(total_rewards_per_episode)
    
    plt.scatter(np.arange(0, nm_episodes), total_rewards_per_episode_arr)
    plt.savefig('ep_d_r_0_001_lr_09_dc_05ddd.png')
    ##############################################################

    # csv_file_path = 'ep_d_r_0_001_lr_alpha_1_dc_0_9.csv'

    # with open(csv_file_path, 'w', newline='') as csv_file:
    # 	csv_writer = csv.writer(csv_file)
    # 	csv_writer.writerow(['state', 'up', 'right', 'down', 'left'])

    # 	for i in range(48):
    # 		round_n = 5
    # 		csv_writer.writerow([i, round(q[i,0],round_n), round(q[i,1],round_n), round(q[i,2],round_n), round(q[i,3], round_n)])

    