# %%
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision import transforms

def load_frames(folder, video_subset=None):
    """Load and preprocess frames from a specified subset of videos."""
    frames = []
    for video_folder in sorted(os.listdir(folder)):
        if video_subset and video_folder not in video_subset:
            continue
        video_path = os.path.join(folder, video_folder)
        if os.path.isdir(video_path):
            for filename in sorted(os.listdir(video_path)):
                filepath = os.path.join(video_path, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    image = cv2.resize(image, (84, 84)) # Resize to match input size
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
                    frames.append(image / 255.0) # Normalize to [0, 1]
    return np.array(frames)

# %%
data_path = "../CPSC 6420 Flame 2 Research/Segmented Frames"
data = {
    "IR": load_frames(os.path.join(data_path, "IR")),
    "RGB": load_frames(os.path.join(data_path, "RGB"))
}

# %%
from sklearn.model_selection import train_test_split

train_data, test_data = {}, {}
for modality in data:
    train, test = train_test_split(data[modality], test_size=0.2, random_state=42)
    train_data[modality] = torch.tensor(train).unsqueeze(1).float()  # Add channel dimension
    test_data[modality] = torch.tensor(test).unsqueeze(1).float()

# %%
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (84, 84) -> (42, 42)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (42, 42) -> (21, 21)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (21, 21) -> (11, 11)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        x = self.conv(state)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)
    
action_dim = 1
actor = Actor(action_dim)
actor_target = Actor(action_dim)
actor_target.load_state_dict(actor.state_dict())

# %%
class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 11 * 11 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = self.conv(state)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, action], dim=1)  # Concatenate state and action
        return self.fc(x)

critic = Critic(action_dim)
critic_target = Critic(action_dim)
critic_target.load_state_dict(critic.state_dict())

# %%
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),  # Convert numpy arrays to tensors
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )


buffer = ReplayBuffer(100000)

# %%
gamma = 0.99  # Discount factor
tau = 0.005   # Soft update factor
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
batch_size = 64

def train_step():
    if len(buffer.buffer) < batch_size:
        return

    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Update Critic
    next_actions = actor_target(next_states)
    target_q = rewards + gamma * (1 - dones) * critic_target(next_states, next_actions).detach()
    current_q = critic(states, actions)
    critic_loss = nn.MSELoss()(current_q, target_q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Update Actor
    actor_loss = -critic(states, actor(states)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# %%
def get_next_state_and_reward(index, data, modality):
    """Simulate environment's next state, reward, and done flag."""
    next_state = data[index + 1] if index + 1 < len(data) else data[index]
    reward = np.random.rand()  # Change this
    done = index + 1 == len(data)
    return next_state, reward, done

episodes = 100 
modality = "RGB"  # IR/RGB
data = train_data[modality]  # Load train data

episode_rewards = []

for episode in range(episodes):
    state = data[0].unsqueeze(0)
    total_reward = 0

    for t in range(len(data)):
        # Get action from the actor
        action = actor(state).detach().numpy().squeeze()

        # Simulate environment step
        next_state, reward, done = get_next_state_and_reward(t, data, modality)
        next_state = next_state.unsqueeze(0)  # Add batch dimension
        total_reward += reward

        # Ensure correct data types and dimensions for ReplayBuffer
        state_np = state.squeeze(0).numpy()  # Convert to numpy and remove batch dimension
        next_state_np = next_state.squeeze(0).numpy()  # Same for next_state
        action_np = np.array([action])
        reward_np = np.array([reward])
        done_np = np.array([done], dtype=np.float32)

        buffer.add(state_np, action_np, reward_np, next_state_np, done_np)

        # Update state
        state = next_state

        # Perform training step
        train_step()

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

# %%
import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance Over Episodes")
plt.show()


