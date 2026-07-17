# Run with `python3 -m examples.train_pendulum_sac`
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

import gymnasium as gym

from src.buffers import ReplayMemory
from src.trainers import SACTrainer


env = gym.make("Pendulum-v1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = SACTrainer(
    env,
    memory=ReplayMemory(memory_size=int(1e5), batch_size=128),
    device=device,
)

n_episodes = 300
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
alpha_learning_rate = 3e-4
gamma = 0.99
tau = 0.005
start_steps = 1000
hidden_size = 256
log_std_min = -20
log_std_max = 2


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, action_low, action_high):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, n_actions)
        self.log_std = nn.Linear(hidden_size, n_actions)

        action_low = torch.tensor(action_low, dtype=torch.float32)
        action_high = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2)
        self.register_buffer("action_bias", (action_high + action_low) / 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(log_std_min, log_std_max)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        dist = Normal(mean, log_std.exp())
        pre_tanh_action = dist.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = tanh_action * self.action_scale + self.action_bias

        log_prob = dist.log_prob(pre_tanh_action)
        log_prob -= torch.log(self.action_scale * (1 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


state, _ = env.reset()
n_observations = len(state)
n_actions = env.action_space.shape[0]

actor = Actor(
    n_observations,
    n_actions,
    action_low=env.action_space.low,
    action_high=env.action_space.high,
).to(device)

critic1 = Critic(n_observations, n_actions).to(device)
critic2 = Critic(n_observations, n_actions).to(device)
critic1_target = Critic(n_observations, n_actions).to(device)
critic2_target = Critic(n_observations, n_actions).to(device)

actor_optimizer = Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = Adam(
    list(critic1.parameters()) + list(critic2.parameters()),
    lr=critic_learning_rate,
)

log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=device, requires_grad=True)
alpha_optimizer = Adam([log_alpha], lr=alpha_learning_rate)

trainer.train(
    actor=actor,
    critic1=critic1,
    critic2=critic2,
    critic1_target=critic1_target,
    critic2_target=critic2_target,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    criterion=nn.MSELoss(),
    n_episodes=n_episodes,
    gamma=gamma,
    tau=tau,
    max_steps=200,
    start_steps=start_steps,
    log_alpha=log_alpha,
    alpha_optimizer=alpha_optimizer,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/pendulum_sac_actor.pt")
torch.save(critic1.state_dict(), "saves/pendulum_sac_critic1.pt")
torch.save(critic2.state_dict(), "saves/pendulum_sac_critic2.pt")

env.close()

env = gym.make("Pendulum-v1", render_mode="human")


def test(model: torch.nn.Module):
    state, _ = env.reset()
    done = False
    model.eval()

    while not done:
        action = trainer.policy(model, state, deterministic=True)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()


test(actor)
env.close()
