import gymnasium as gym
import numpy as np
from discretization import find_closest
import pickle
from typing import  Tuple


class RLDiscretizationParams:
    def __init__(self,
                 position_count: int,
                 velocity_count: int,
                 force_count: int,
                 ) -> None:
        self.position_count = position_count
        self.velocity_count = velocity_count
        self.force_count = force_count

class ValueWithIndex:
    def __init__(self, value: float, index: int) -> None:
        self.value = value
        self.index = index


class RLWithDiscretization:
    def __init__(
        self,
        rl_discretization_params : RLDiscretizationParams,
        learning_rate: float = 0.9,
        discount_coeff: float = 0.9,
        render=False
    ) -> None:
        self.env = gym.make('MountainCarContinuous-v0', render_mode='human' if render else None)
        self.discount_coeff = discount_coeff
        self.learning_rate = learning_rate
        self.sarsa_discretization_params = rl_discretization_params
        self.q_table = np.zeros((rl_discretization_params.position_count, 
                                    rl_discretization_params.velocity_count, 
                                    rl_discretization_params.force_count
                                ))
        self.available_positions, self.available_velocities, self.available_forces = self.get_available_states_and_actions()
        self.rewards = []
        self.terminated_episodes = []
        self.iterations = []


    def get_available_states_and_actions(self) -> Tuple:
        available_positions = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], self.sarsa_discretization_params.position_count)
        available_velocities = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], self.sarsa_discretization_params.velocity_count)
        available_forces = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0], self.sarsa_discretization_params.force_count)
        available_forces = [np.array([elem], dtype=np.float32) for elem in available_forces]

        return available_positions, available_velocities, available_forces
    
    def get_init_state(self):
        state = self.env.reset()[0]      # Starting position, starting velocity always 0
        state_position = np.digitize(state[0], self.available_positions)
        state_velocity = np.digitize(state[1], self.available_velocities)
        return state, state_position, state_velocity

    def get_random_action(self):
        action = self.env.action_space.sample()
        closest_action = find_closest(self.available_forces, action)
        return closest_action

    def load_q_table_from_file(self, filename):
        f = open(filename, 'rb')
        self.q_table = pickle.load(f)
        f.close()

    def run(self, episodes, is_training=False):
        rng = np.random.default_rng()
        epsilon = 1
        epsilon_decay_rate = 1.2/episodes 
        for _ in range(episodes):
            self.run_single_episode(epsilon, is_training, rng=rng)
            epsilon = max(epsilon - epsilon_decay_rate, 0)
        
    def run_single_episode(self, epsilon, is_training, rng=np.random.default_rng()):
        raise NotImplementedError("Implement the run_single_episode method")

    def save_q_table_to_file(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.q_table, f)
        f.close()

    def save_rewards_to_file(self, filename):
        data = {
            'rewards': self.rewards,
            'iterations': self.iterations,
            'terminated': self.terminated_episodes
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    

def load_rewards_from_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data