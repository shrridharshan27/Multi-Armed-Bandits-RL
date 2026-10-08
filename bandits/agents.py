"""
Multi-Armed Bandit Agents: Epsilon-Greedy, UCB, and Thompson Sampling.
"""
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, num_actions: int, eps: float = 0.1):
        self.num_actions = num_actions
        self.eps = eps
        self.pull_count = np.zeros(num_actions, dtype=int)
        self.q_values = np.zeros(num_actions, dtype=float)

    def choose_action(self) -> int:
        if np.random.rand() < self.eps:
            return int(np.random.randint(0, self.num_actions))
        return int(np.argmax(self.q_values))

    def update(self, action: int, reward: float) -> None:
        self.pull_count[action] += 1
        n = self.pull_count[action]
        self.q_values[action] += (reward - self.q_values[action]) / n

class UCBandit:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.pull_count = np.zeros(num_actions, dtype=int)
        self.q_values = np.zeros(num_actions, dtype=float)
        self.step = 0

    def choose_action(self) -> int:
        # Initial exploration: try each arm once
        if self.step < self.num_actions:
            return self.step
        bonus = np.sqrt(2 * np.log(self.step) / (self.pull_count + 1e-9))
        return int(np.argmax(self.q_values + bonus))

    def update(self, action: int, reward: float) -> None:
        self.pull_count[action] += 1
        self.step += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.pull_count[action]

class ThompsonBandit:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        # Prior Beta distribution parameters
        self.alpha = np.ones(num_actions, dtype=float)
        self.beta = np.ones(num_actions, dtype=float)

    def choose_action(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
