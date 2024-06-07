from typing import Any, List, Callable
import numpy as np
import scipy as sp


class SarsaWithAproximation:
    def __init__(
        self,
        aproximation_functions: List[Callable[[float, float], float]],
        learning_rate: float = 1,
        epsilon: float = 0.9,
        gamma: float = 0.9,
    ) -> None:
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # jak bardzo patrzy w przód
        self.approximation_functions = aproximation_functions
        self.q_table = np.array([0 for _ in range(len(aproximation_functions))])

    def get_aproximation_vector(self, state: float, action: float) -> np.ndarray:
        return np.array([f(state, action) for f in self.approximation_functions])

    def __call__(self, state: float, action: int) -> Any:
        return self.q_table @ self.get_aproximation_vector(state, action)

    def get_best_action(self, state: float) -> int:
        return sp.optimize.fminbound(lambda x: -self(state, x), -1, 1)

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_action: np.ndarray,
    ) -> None:
        temporal_difference = (
            reward
            + self.gamma * self(next_state, next_action.item())
            - self(state, action.item())
        )
        gt = self.get_aproximation_vector(state, action.item()) * temporal_difference
        updatet_q = self.q_table + self.learning_rate * gt

        self.q_table = updatet_q

    def save_to_file(self, filename: str) -> None:
        np.save(filename, self.q_table)

    def load_from_file(self, filename: str) -> None:
        self.q_table = np.load(filename)


if __name__ == "__main__":
    test = SarsaWithAproximation(
        [lambda x, y: x[0] + x[1] + y, lambda x, y: x[1] * x[0] * y]
    )
    test.load_from_file("test.npy")
    print(test([2, 3], 2))
