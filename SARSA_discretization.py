import numpy as np
import matplotlib.pyplot as plt
from RL_discretization import RLWithDiscretization, RLDiscretizationParams
#


class SarsaWithDiscretization(RLWithDiscretization):
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
        if is_training and rng.random() < epsilon:
            action = self.get_random_action()
        else:
            action_index = np.argmax(self.q_table[position_state_i, velocity_state_i, :])
            action = self.available_forces[action_index]
        action_index = self.available_forces.index(action)

        while(not terminated and rewards>-2000):
            new_state,reward,terminated,_,_ = self.env.step(action)
            new_position_state_i = np.digitize(new_state[0], self.available_positions)
            new_velocity_state_i = np.digitize(new_state[1], self.available_velocities)

            if is_training and rng.random() < epsilon:
                next_action = self.get_random_action()
            else:
                next_action_index = np.argmax(self.q_table[new_position_state_i, new_velocity_state_i, :])
                next_action = self.available_forces[next_action_index]

            next_action_index = self.available_forces.index(next_action)

            if is_training:
                self.q_table[position_state_i, velocity_state_i, action_index] = self.q_table[position_state_i, velocity_state_i, action_index] + ...
                self.learning_rate * (
                    reward + self.discount_coeff * self.q_table[new_position_state_i, new_velocity_state_i, next_action_index] - ...
                    - self.q_table[position_state_i, velocity_state_i, action_index]
                )

            position_state_i = new_position_state_i
            velocity_state_i = new_velocity_state_i
            reward -= 0.5
            rewards += reward
            num_iteration += 1
            action = next_action
            action_index = next_action_index
        self.rewards.append(rewards)
        self.terminated_episodes.append(terminated)
        self.iterations.append(num_iteration)



sarsa_params = RLDiscretizationParams(
    position_count=2,
    velocity_count=10,
    force_count=10
)

learning_rate=0.9
discount_coeff=0.5
sarsa = SarsaWithDiscretization(
    rl_discretization_params=sarsa_params,
    learning_rate=learning_rate,
    discount_coeff=discount_coeff,
    render=False
)

sarsa.run(100, True)
sarsa.save_q_table_to_file(f's/q_table_disc_{discount_coeff}_learn{learning_rate}.pkl')
sarsa.save_rewards_to_file(
    f'SARSA/new_rewards/mountain_car_cont_f_{sarsa_params.force_count}_v_{sarsa_params.velocity_count}_p_{sarsa_params.position_count}_train.pkl'
    # f's/rewards_disc_{discount_coeff}_learn{learning_rate}.pkl'
)

# sarsa2 = SarsaWithDiscretization(
#     rl_discretization_params=sarsa_params,
#     learning_rate=0.8,
#     discount_coeff=0.9,
#     render=True
# )
# sarsa2.load_q_table_from_file('SARSA/QTables/mountain_car_cont_f_4_v_28_p_28.pkl')
# sarsa2.run(1000)
# data = load_rewards_from_file(f'SARSA/new_rewards/mountain_car_cont_f_{sarsa_params.force_count}_v_{sarsa_params.velocity_count}_p_{sarsa_params.position_count}_train.pkl')