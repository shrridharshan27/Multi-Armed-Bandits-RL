"""
Multi-Armed Bandit Comparison
Epsilon-Greedy vs Upper Confidence Bound (UCB) vs Thompson Sampling
Shrri Dharshan D R - 23BPS1090
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from bandits.environment import MultiArmedBanditEnv
from bandits.agents import EpsilonGreedyBandit, UCBandit, ThompsonBandit

def run_simulation(num_steps: int = 1000, num_actions: int = 10, eps: float = 0.1):
    env = MultiArmedBanditEnv(num_actions=num_actions, seed=42)
    print(f"True reward means per arm: {np.round(env.arm_means, 2)}")
    print(f"Optimal arm: #{env.optimal_arm} with mean reward = {env.arm_means[env.optimal_arm]:.2f}\n")

    # Epsilon-Greedy
    eg_agent = EpsilonGreedyBandit(num_actions, eps)
    eg_rewards, eg_regrets = [], []
    eg_total = 0.0

    for step in range(num_steps):
        a = eg_agent.choose_action()
        r = env.step(a)
        eg_agent.update(a, r)
        eg_total += r
        eg_rewards.append(eg_total)
        eg_regrets.append((step + 1) * env.arm_means[env.optimal_arm] - eg_total)

    print(f"Epsilon-Greedy (eps={eps}) Total Reward: {eg_total:.2f}")

    # UCB
    ucb_agent = UCBandit(num_actions)
    ucb_rewards, ucb_regrets = [], []
    ucb_total = 0.0

    for step in range(num_steps):
        a = ucb_agent.choose_action()
        r = env.step(a)
        ucb_agent.update(a, r)
        ucb_total += r
        ucb_rewards.append(ucb_total)
        ucb_regrets.append((step + 1) * env.arm_means[env.optimal_arm] - ucb_total)

    print(f"UCB Total Reward: {ucb_total:.2f}")

    # Thompson Sampling
    ts_agent = ThompsonBandit(num_actions)
    ts_rewards, ts_regrets = [], []
    ts_total = 0.0

    for step in range(num_steps):
        a = ts_agent.choose_action()
        r = env.step(a)
        ts_agent.update(a, r)
        ts_total += r
        ts_rewards.append(ts_total)
        ts_regrets.append((step + 1) * env.arm_means[env.optimal_arm] - ts_total)

    print(f"Thompson Sampling Total Reward: {ts_total:.2f}")

    # Plot 1: Cumulative Reward
    plt.figure(figsize=(10, 5))
    plt.plot(eg_rewards, label=f'Epsilon-Greedy (eps={eps})', color='#2b5c8f', lw=2)
    plt.plot(ucb_rewards, label='Upper Confidence Bound (UCB)', color='#d9534f', lw=2)
    plt.plot(ts_rewards, label='Thompson Sampling', color='#5cb85c', lw=2)
    plt.title('Multi-Armed Bandit: Cumulative Reward over Steps', fontsize=13, fontweight='bold')
    plt.xlabel('Steps', fontsize=11)
    plt.ylabel('Cumulative Reward', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot1_path = os.path.join("assets", "cumulative_rewards.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()

    # Plot 2: Cumulative Regret
    plt.figure(figsize=(10, 5))
    plt.plot(eg_regrets, label=f'Epsilon-Greedy (eps={eps})', color='#2b5c8f', lw=2)
    plt.plot(ucb_regrets, label='Upper Confidence Bound (UCB)', color='#d9534f', lw=2)
    plt.plot(ts_regrets, label='Thompson Sampling', color='#5cb85c', lw=2)
    plt.title('Multi-Armed Bandit: Cumulative Regret over Steps', fontsize=13, fontweight='bold')
    plt.xlabel('Steps', fontsize=11)
    plt.ylabel('Cumulative Regret', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot2_path = os.path.join("assets", "regret_comparison.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()

    print(f"Plots saved to {plot1_path} and {plot2_path}")

if __name__ == "__main__":
    run_simulation()
