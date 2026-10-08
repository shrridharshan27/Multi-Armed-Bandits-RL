"""
Multi-Armed Bandit Environment
Simulates a multi-armed bandit problem with stationary Gaussian rewards.
"""
import numpy as np
from typing import Tuple

class MultiArmedBanditEnv:
    def __init__(self, num_actions: int = 10, seed: int = 42):
        self.num_actions = num_actions
        np.random.seed(seed)
        self.arm_means = np.random.randn(num_actions)
        self.optimal_arm = int(np.argmax(self.arm_means))

    def step(self, action: int) -> float:
        """Pull arm and observe reward sampled from normal distribution."""
        if not (0 <= action < self.num_actions):
            raise ValueError(f"Action {action} out of bounds.")
        reward = np.random.randn() + self.arm_means[action]
        return float(reward)
