import os
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
#from .models import Actor, Critic
from utils import compute_advantages, save_hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.mean = nn.Linear(64, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = torch.tanh(self.mean(x))
        return mean, self.log_std.exp()

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value(x)

class PPOAgent:
    def __init__(self, env, gamma=0.99, lamda=0.95, clip_param=0.2):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        # Khởi tạo Actor và Critic
        self.actor = Actor(self.state_size, self.action_size).to(device)
        self.critic = Critic(self.state_size).to(device)

        # Khởi tạo optimizer cho Actor và Critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Bộ nhớ để lưu các trải nghiệm
        self.memory = deque(maxlen=50000)
        self.writer = SummaryWriter()  # TensorBoard Writer
        self.gamma = gamma
        self.clip_param = clip_param
        self.lamda = lamda
        save_hyperparameters(self.writer, {'gamma': gamma, 'lamda': lamda, 'clip_param': clip_param})

        self.max_average_score = -float('inf')  # Dùng cho Early Stopping

        # Kiểm tra và tải mô hình nếu đã có
        if os.path.exists("ppo_actor_hardcore.pth") and os.path.exists("ppo_critic_hardcore.pth"):
            self.load()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy().flatten(), log_prob.cpu().item()

    def store_transition(self, state, action, reward, done, next_state, log_prob):
        self.memory.append((state, action, reward, done, next_state, log_prob))

    def update(self, episode):
        states, actions, rewards, dones, next_states, log_probs = zip(*self.memory)
        self.memory.clear()

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        log_probs = torch.FloatTensor(log_probs).to(device)

        values = self.critic(states).detach().squeeze()
        next_value = self.critic(torch.FloatTensor(next_states).to(device)).detach().squeeze()
        # print("Rewards:", rewards)
        # print("Dones:", dones)
        # print("Values:", values)
        # print("Next Value:", next_value)

        advantages = compute_advantages(rewards, dones, values, next_value)

        returns = advantages + values

        # Actor Update
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - log_probs)

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Critic Update
        values = self.critic(states).squeeze()
        critic_loss = torch.nn.MSELoss()(values, returns)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Ghi log vào TensorBoard
        self.writer.add_scalar('Loss/Actor_hardcore', actor_loss.item(), episode)
        self.writer.add_scalar('Loss/Critic_hardcore', critic_loss.item(), episode)

    def save(self):
        """Lưu trọng số của Actor và Critic."""
        torch.save(self.actor.state_dict(), "ppo_actor_hardcore.pth")
        torch.save(self.critic.state_dict(), "ppo_critic_hardcore.pth")
        print("Model saved successfully.")

    def load(self):
        """Tải trọng số đã lưu của Actor và Critic."""
        self.actor.load_state_dict(torch.load("ppo_actor_hardcore.pth", weights_only=True))
        self.critic.load_state_dict(torch.load("ppo_critic_hardcore.pth", weights_only=True))
        print("Model loaded successfully.")

    def train(self, episodes=1000):
        scores = []

        for episode in range(episodes):
            state,_ = self.env.reset()
            #print(f"Initial state: {state}, Length: {len(state) if state is not None else 'None'}")
            score = 0
            done = False

            while not done:
                if state is None or len(state) != self.state_size:
                    print(f"Invalid state encountered: {state}")
                    break 
                
                action, log_prob = self.select_action(state)
                #print(f"Action: {action}, Shape: {action.shape}")
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.store_transition(state, action, reward, done, next_state, log_prob)

                state = next_state
                score += reward

                if len(self.memory) >= 64:
                    self.update(episode)

            scores.append(score)
            avg_score = np.mean(scores[-100:])  # Trung bình 100 tập gần nhất
            print(f'Episode {episode}, Score: {score}, Average Score: {avg_score}')
            
            # Ghi log điểm số vào TensorBoard
            self.writer.add_scalar('Score_hardcore', score, episode)
            self.writer.add_scalar('Average_Score_hardcore', avg_score, episode)

            # Lưu mô hình sau mỗi tập
            self.save()

            if avg_score > self.max_average_score:
                self.max_average_score = avg_score

            if avg_score >= 250:  # Early Stopping
                print("Training Complete. Average score >= 300.")
                break

        torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
        self.env.close()

    def test(self, test_episodes=10):
        """Kiểm tra hiệu suất của mô hình với các tập thử nghiệm."""
        self.load()
        total_score = 0

        self.env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")  # Khởi tạo lại với chế độ render

        for episode in range(test_episodes):
            state, _ = self.env.reset()
            score = 0
            done = False

            while not done:
                self.env.render()

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                mean, _ = self.actor(state_tensor)
                action = mean.cpu().detach().numpy().flatten()  # Chuyển thành mảng 1 chiều

                # Kiểm tra kích thước của action
                if action.shape != (4,):
                    print(f"Invalid action shape: {action.shape}, expected (4,)")
                    break

                next_state, reward, done, truncated, _ = self.env.step(action)

                state = next_state
                score += reward

            print(f"Test Episode {episode + 1}, Score: {score}")
            total_score += score

        avg_score = total_score / test_episodes
        print(f"Average Score over {test_episodes} episodes: {avg_score}")

        self.env.close()


if __name__ == "__main__":
    env = gym.make("BipedalWalkerHardcore-v3")
    agent = PPOAgent(env)

    agent.train(episodes=10000)
    #agent.test(test_episodes=1)
