# 再実行（コードがリセットされたため再定義）
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# ハイパーパラメータ
STATE_DIM = 2
ACTION_SPACE = [-5.0, 5.0]
ACTION_DIM = len(ACTION_SPACE)
EPISODES = 500
MAX_STEPS = 50
GAMMA = 0.95
EPSILON = 0.1
ALPHA = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 10

class OneDOFArmEnv:
    def __init__(self):
        self.goal = 1.0
        self.reset()

    def reset(self):
        self.theta = np.random.uniform(0, 180)
        return self.get_state()

    def step(self, action_idx):
        action = ACTION_SPACE[action_idx]
        self.theta = np.clip(self.theta + action, 0, 180)
        x_tip = np.cos(np.radians(self.theta))
        reward = -abs(x_tip - self.goal)
        done = False
        return self.get_state(), reward, done

    def get_state(self):
        return np.array([self.theta / 180.0, self.goal])

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self):
        self.q_net = QNetwork(STATE_DIM, ACTION_DIM)
        self.target_net = QNetwork(STATE_DIM, ACTION_DIM)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=ALPHA)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()
        self.steps = 0

    def select_action(self, state):
        if np.random.rand() < EPSILON:
            return np.random.randint(ACTION_DIM)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def store_experience(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + GAMMA * next_q * (~dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# 学習ループ
env = OneDOFArmEnv()
agent = DQNAgent()

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for t in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward

    if ep % 50 == 0:
        print(f"Episode {ep}, Total Reward: {total_reward:.2f}")


def evaluate_agent(env, agent, episodes=5):
    all_trajectories = []
    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        trajectory = [env.theta]
        total_reward = 0

        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            trajectory.append(env.theta)
            total_reward += reward
            state = next_state
            if done:
                break

        all_trajectories.append(trajectory)
        all_rewards.append(total_reward)

    return all_trajectories, all_rewards

# 評価して描画
trajectories, rewards = evaluate_agent(env, agent)

plt.figure(figsize=(10, 4))
for i, traj in enumerate(trajectories):
    x_pos = [np.cos(np.radians(theta)) for theta in traj]
    plt.plot(x_pos, marker='o', label=f"Episode {i+1}")
plt.axhline(y=env.goal, color='r', linestyle='--', label='Goal')
plt.title("End-effector Trajectories After DQN Training")
plt.xlabel("Step")
plt.ylabel("X position of end-effector")
plt.grid(True)
plt.legend()
plt.show()

print(f"Average reward over {len(rewards)} test episodes: {np.mean(rewards):.2f}")
