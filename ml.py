import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import math
from gymnasium import spaces

# Custom environment for the jumping task
class JumpEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: no jump, 1: jump
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # x, y, z positions and velocity
            high=np.array([10, 5, 10, 5]),
            dtype=np.float32
        )
        
        # Environment parameters
        self.gravity = -9.81
        self.jump_force = 7.0
        self.forward_speed = 2.0
        self.dt = 0.05
        self.fence_position = 5.0
        self.fence_height = 1.5
        
        # State variables
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.time = 0.0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.time = 0.0
        return self._get_observation(), {}
    
    def _get_observation(self):
        return np.array([
            self.position[0],  # x position
            self.position[1],  # y position (height)
            self.fence_position - self.position[0],  # distance to fence
            self.velocity[1]  # vertical velocity
        ])
    
    def step(self, action):
        # Apply action (jump)
        if action == 1 and self.position[1] == 0:  # Can only jump from ground
            self.velocity[1] = self.jump_force
        
        # Update physics
        self.velocity[1] += self.gravity * self.dt
        self.position[0] += self.forward_speed * self.dt
        self.position[1] += self.velocity[1] * self.dt
        
        # Ground collision
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = 0
            
        self.time += self.dt
        
        # Check if episode is done
        done = False
        reward = 0
        
        # Collision with fence
        if (abs(self.position[0] - self.fence_position) < 0.2 and 
            self.position[1] < self.fence_height):
            done = True
            reward = -1
        
        # Successfully cleared fence
        if self.position[0] > self.fence_position + 0.2:
            done = True
            reward = 1
            
        # Out of bounds
        if self.position[0] > self.fence_position + 2:
            done = True
        
        return self._get_observation(), reward, done, False, {}

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.target_update = 10
        self.memory = ReplayMemory(10000)
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.steps_done = 0
        
    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state).max(1)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute max_a Q(s_{t+1}, a) for all next states
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values
        
        # Compute loss
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Training loop
def train():
    env = JumpEnvironment()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    episodes = 1000
    best_reward = -float('inf')
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        losses = []
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            loss = agent.optimize_model()
            if loss is not None:
                losses.append(loss)
            
            if done:
                break
        
        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        # Save the model if it achieves better performance
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), 'best_jump_model.pth')
            print(f"New best model saved with reward: {best_reward}")
        
        # Save checkpoint every 100 episodes
        if episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'checkpoint_model_episode_{episode}.pth')
            print(f"Checkpoint saved at episode {episode}")
        
        if episode % 10 == 0:
            avg_loss = np.mean(losses) if losses else 0
            print(f"Episode {episode}, Total Reward: {total_reward}, Average Loss: {avg_loss:.4f}")
    
    # Save final model
    torch.save(agent.policy_net.state_dict(), 'final_jump_model.pth')
    print("Final model saved as 'final_jump_model.pth'")
            
    return agent

if __name__ == "__main__":
    trained_agent = train()
    