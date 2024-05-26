from typing import Any, List, Callable
import numpy as np
import scipy as sp


class SarsaWithAproximation:
    def __init__(
        self,
        aproximation_functions: List[Callable[[float, float], float]],
        learning_rate: float = 1,
        gamma: float = 0.9,
    ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma  # jak bardzo patrzy w przód
        self.approximation_functions = aproximation_functions
        self.q_table = np.array([1 for _ in range(len(aproximation_functions))])

    def get_aproximation_vector(self, state: float, action: float) -> np.ndarray:
        return [f(state, action) for f in self.approximation_functions]

    def __call__(self, state: float, action: int) -> Any:
        return self.q_table @ self.get_aproximation_vector(state, action)

    def get_best_action(self, state: float) -> int:
        return sp.optimize.fminbound(lambda x: -self(state, x), -1, 1)

    def update(
        self,
        state: float,
        action: int,
        reward: float,
        next_state: float,
        next_action: int,
    ) -> None:
        temporal_difference = (
            reward + self.gamma * self(next_state, next_action) - self(state, action)
        )
        gt = self.get_aproximation_vector(state, action) * temporal_difference
        updatet_q = self.q_table + self.learning_rate * gt

        self.q_table = updatet_q


if __name__ == "__main__":
    test = SarsaWithAproximation([lambda x, y: x + y, lambda x, y: x * y])
    print(test(1, 2))
