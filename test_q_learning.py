from itertools import product
from typing import Any, Tuple

import gym

from q_learning import BaseEnvironment, QLearner


class Environment(BaseEnvironment):
    def __init__(self) -> None:
        self.allowed = frozenset(
            {
                (0, 1),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 1),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 1),
                (4, 1),
            }
        )

    def reset(self) -> Tuple[int, int]:
        self.current_state = (2, 0)
        return self.current_state

    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool, Any]:
        r, c = self.current_state
        if action == "↑":
            r -= 1
        elif action == "→":
            c += 1
        elif action == "↓":
            r += 1
        elif action == "←":
            c -= 1
        else:
            raise ValueError("Invalid action")
        if 0 <= r < 5 and 1 <= c < 6 and (r, c) in self.allowed:
            self.current_state = (r, c)
            return (r, c), -1, False, None
        elif (r, c) == (2, 6):
            self.current_state = (r, c)
            return (r, c), -1, True, None
        else:
            return self.current_state, -1, False, None


def test_qlearn() -> None:
    actions = ["↑", "→", "↓", "←"]
    states = [(i, j) for i, j in product(range(5), range(1, 6))] + [(2, 0), (2, 6)]

    environment = Environment()

    q_learner: QLearner[Tuple[int, int], str] = QLearner(
        actions=actions, states=states, environment=environment, n_iter=1000
    )
    final_reward = q_learner.learn()
    assert int(final_reward) == -6
    print(q_learner)


def test_frozen() -> None:
    env = gym.make("FrozenLake-v0")
    q_learner = QLearner(
        actions=[0, 1, 2, 3], states=list(range(16)), environment=env, n_iter=10000
    )
    final_reward = q_learner.learn()
    print(final_reward, q_learner)


if __name__ == "__main__":
    test_qlearn()
    test_frozen()
