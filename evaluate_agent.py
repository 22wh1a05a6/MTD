import numpy as np
import matplotlib.pyplot as plt
import random
from env.mtd_env import MTDEnv
from stable_baselines3 import PPO
# Parameters

STEPS = 50
EPISODES = 3
EPSILON = 0.1

ATTACK_THRESHOLD = 0.7
# Create Environment
env = MTDEnv()
# Baseline Evaluation
baseline_syn = []
baseline_attack = 0

print("\nRunning BASELINE (No Defense)")

for ep in range(EPISODES):

    state, _ = env.reset()

    for step in range(STEPS):

        action = 0
        state, reward, done, _, _ = env.step(action)

        syn_rate = state[0]
        baseline_syn.append(syn_rate)

        if syn_rate > ATTACK_THRESHOLD:
            baseline_attack += 1

        if done:
            break

print("Baseline Mean SYN Rate:", np.mean(baseline_syn))


# Load PPO Model

model = PPO.load("results/ppo_mtd_improved.zip")

# PPO Evaluation

ppo_syn = []
ppo_actions = []
ppo_rewards = []
ppo_step_rewards = []

ppo_attack = 0

print("\nRunning PPO Agent")

for ep in range(EPISODES):

    state, _ = env.reset()
    episode_reward = 0

    for step in range(STEPS):

        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(state, deterministic=True)
            action = int(action)

        state, reward, done, _, _ = env.step(action)

        syn_rate = state[0]

        ppo_syn.append(syn_rate)
        ppo_actions.append(action)

        episode_reward += reward
        ppo_step_rewards.append(reward)

        if syn_rate > ATTACK_THRESHOLD:
            ppo_attack += 1

        if done:
            break

    ppo_rewards.append(episode_reward)

print("PPO Mean SYN Rate:", np.mean(ppo_syn))


# SYN Rate Comparison Plot
plt.figure(figsize=(10,5))

plt.plot(baseline_syn, label="Baseline")
plt.plot(ppo_syn, label="PPO MTD")

plt.xlabel("Steps")
plt.ylabel("SYN Rate")
plt.title("SYN Rate Comparison (Baseline vs PPO)")
plt.legend()

plt.savefig("results/syn_rate_comparison_27.png")
plt.show()

# Attack Reduction Metrics
baseline_mean = np.mean(baseline_syn)
ppo_mean = np.mean(ppo_syn)

attack_reduction = (baseline_mean - ppo_mean) * 100
mitigation = ((baseline_mean - ppo_mean) / baseline_mean) * 100

print("\n==============================")
print("Evaluation Result")
print("==============================")
print("Baseline Attack Level:", baseline_mean)
print("PPO Attack Level:", ppo_mean)
print("Attack Reduction (%):", attack_reduction)
print("Mitigation (%):", mitigation)

# Attack Mitigation Plot

plt.figure()

plt.bar(["Attack Mitigation"], [mitigation])

plt.ylabel("Reduction (%)")
plt.title("Attack Traffic Reduction using PPO-MTD")

plt.ylim(0,100)

plt.text(0, mitigation + 1, f"{mitigation:.1f}%", ha="center")

plt.savefig("results/attack_mitigation_27.png")
plt.show()


# MTD Action Distribution
plt.figure()

actions = [0,1,2,3]
counts = [ppo_actions.count(a) for a in actions]

plt.bar(actions, counts)

plt.xlabel("MTD Action")
plt.ylabel("Frequency")
plt.title("MTD Action Distribution")

plt.xticks(actions)

plt.savefig("results/mtd_action_distribution_27.png")
plt.show()

# Reward Smoothing Function
def moving_avg(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# PPO Reward Curve
plt.figure()

smoothed_rewards = moving_avg(ppo_rewards)

plt.plot(smoothed_rewards, label="Smoothed Reward", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Reward Curve")

plt.legend()

plt.savefig("results/ppo_reward_curve_27.png")
plt.show()


# Step-wise Reward Plot 
plt.figure()

plt.plot(ppo_step_rewards)

plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Step-wise PPO Reward")

plt.savefig("results/ppo_step_reward.png")
plt.show()
