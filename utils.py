import torch
import numpy as np

def compute_advantages(rewards, dones, values, next_value, gamma=0.99, lamda=0.95):
    
    advantages = torch.zeros_like(values)  # Khởi tạo tensor chứa lợi thế
    advantage = 0 
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value[t] * (1 - dones[t]) - values[t]
        advantage = delta + gamma * lamda * advantage * (1 - dones[t])
        advantages[t] = advantage
    
    return advantages

def save_hyperparameters(writer, params):
    for key, value in params.items():
        writer.add_text(f'Hyperparameter/{key}', str(value))
