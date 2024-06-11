import numpy as np
from RL_discretization import RLWithDiscretization, RLDiscretizationParams

class QLearningWithDiscretization(RLWithDiscretization):
    def __init__(
        self,
        rl_discretization_params : RLDiscretizationParams,
        learning_rate: float = 0.9,
        discount_coeff: float = 0.9,
        render=False
    ) -> None:
        super().__init__(rl_discretization_params=rl_discretization_params, learning_rate=learning_rate, discount_coeff=discount_coeff,render=render)

    def run_single_episode(self, epsilon, is_training, rng=np.random.default_rng()):
        terminated = False
        rewards = 0
        _, position_state_i, velocity_state_i = self.get_init_state()
        num_iteration = 0
        while(not terminated and rewards>-2000):
            if is_training and rng.random() < epsilon:
                action = self.get_random_action()
            else:
                action_index = np.argmax(self.q_table[position_state_i, velocity_state_i, :])
                action = self.available_forces[action_index]

            action_index = self.available_forces.index(action)
            new_state,reward,terminated,_,_ = self.env.step(action)
            new_position_state_i = np.digitize(new_state[0], self.available_positions)
            new_velocity_state_i = np.digitize(new_state[1], self.available_velocities)

            if is_training:
                self.q_table[position_state_i, velocity_state_i, action_index] = self.q_table[position_state_i, velocity_state_i, action_index] + self.learning_rate * (
                    reward + self.discount_coeff * np.max(self.q_table[new_position_state_i, new_velocity_state_i, :]) - self.q_table[position_state_i, velocity_state_i, action_index]
                )

            position_state_i = new_position_state_i
            velocity_state_i = new_velocity_state_i
            reward -= 0.5
            rewards += reward
            num_iteration += 1
            _ = new_state
        
        self.rewards.append(rewards)
        self.terminated_episodes.append(terminated)
        self.iterations.append(num_iteration)