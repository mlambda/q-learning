from io import StringIO
from random import choice as random_choice
from random import random
from typing import Any, Generic, Hashable, Sequence, Tuple, TypeVar

from numpy import float32, linspace, zeros

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


class BaseEnvironment(Generic[StateT, ActionT]):
    def step(self, action: ActionT) -> Tuple[StateT, float, bool, Any]:
        raise NotImplementedError()

    def reset(self) -> StateT:
        raise NotImplementedError()


class QLearner(Generic[StateT, ActionT]):
    def __init__(
        self,
        actions: Sequence[ActionT],
        states: Sequence[StateT],
        environment: BaseEnvironment[StateT, ActionT],
        gamma: float = 0.9,
        alpha: float = 0.1,
        n_iter: int = 100,
    ):
        self.actions = actions
        self.action_to_index = {a: i for i, a in enumerate(actions)}
        self.states = states
        self.state_to_index = {s: i for i, s in enumerate(states)}
        self.environment = environment
        self.gamma = gamma
        self.alpha = alpha
        self.n_iter = n_iter
        self.q_table = zeros((len(states), len(actions)), dtype=float32)

    def __str__(self) -> str:
        with StringIO() as string_io:
            actions_str = " | ".join(f"{str(a):^5}" for a in self.actions)
            string_io.write(f"{' ' * 10} | {actions_str}\n")
            for state, row in zip(self.states, self.q_table):
                state_values = " | ".join(f"{v:5.2f}" for v in row)
                string_io.write(f"{str(state):10} | {state_values}\n")
            return string_io.getvalue()

    def learn(self) -> float:
        len_i = len(str(self.n_iter))
        for i, epsilon in enumerate(linspace(1, 0, self.n_iter), start=1):
            self.epsilon = epsilon
            reward = self.episode()
            print(f"iteration {i:{len_i}}, ε = {epsilon:.2f}, r = {reward:7.2f}")
        return reward

    def get_q(self, state: StateT, action: ActionT) -> float:
        return self.q_table[self.state_to_index[state], self.action_to_index[action]]

    def set_q(self, state: StateT, action: ActionT, value: float) -> None:
        self.q_table[self.state_to_index[state], self.action_to_index[action]] = value

    def bellman_update(self, s: StateT, a: ActionT, r: float, new_s: StateT) -> None:
        q = self.get_q(s, a)
        best_new_s_a = self.best_action(new_s)
        new_s_max_q = self.get_q(new_s, best_new_s_a)
        self.set_q(s, a, q + self.alpha * (r + self.gamma * new_s_max_q - q))

    def best_action(self, s: StateT) -> ActionT:
        _, a = max((self.get_q(s, a), a) for a in self.actions)
        return a

    def episode(self) -> float:
        s = self.environment.reset()
        total_r = 0.0
        done = False
        while not done:
            if random() >= self.epsilon:
                # Exploitation
                a = self.best_action(s)
            else:
                # Exploration
                a = random_choice(self.actions)
            new_s, r, done, _ = self.environment.step(a)
            total_r += r
            self.bellman_update(s, a, r, new_s)
            s = new_s
        return total_r
