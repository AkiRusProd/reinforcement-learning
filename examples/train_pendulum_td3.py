# Run with `python3 -m examples.train_pendulum_td3`
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import gymnasium as gym

from src.buffers import ReplayMemory
from src.noise import OUActionNoise
from src.trainers import TD3Trainer


env = gym.make("Pendulum-v1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = TD3Trainer(
    env,
    memory=ReplayMemory(memory_size=int(1e5), batch_size=128),
    device=device,
)

n_episodes = 300
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
gamma = 0.99
tau = 0.005
policy_delay = 2
target_noise_std = 0.2
target_noise_clip = 0.5
start_steps = 1000


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, scale):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        self.scale = scale

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return torch.tanh(self.layer3(x)) * self.scale


class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


state, _ = env.reset()
n_observations = len(state)
n_actions = env.action_space.shape[0]
action_scale = float(env.action_space.high[0])

actor = Actor(n_observations, n_actions, action_scale).to(device)
actor_target = Actor(n_observations, n_actions, action_scale).to(device)

critic1 = Critic(n_observations, n_actions).to(device)
critic2 = Critic(n_observations, n_actions).to(device)
critic1_target = Critic(n_observations, n_actions).to(device)
critic2_target = Critic(n_observations, n_actions).to(device)

actor_optimizer = Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = Adam(
    list(critic1.parameters()) + list(critic2.parameters()),
    lr=critic_learning_rate,
)

criterion = nn.MSELoss()
exploration_noise = OUActionNoise(mu=np.zeros(n_actions), sigma=0.1)

trainer.train(
    actor=actor,
    critic1=critic1,
    critic2=critic2,
    actor_target=actor_target,
    critic1_target=critic1_target,
    critic2_target=critic2_target,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    criterion=criterion,
    noise=exploration_noise,
    n_episodes=n_episodes,
    gamma=gamma,
    tau=tau,
    max_steps=200,
    policy_delay=policy_delay,
    target_noise_std=target_noise_std,
    target_noise_clip=target_noise_clip,
    start_steps=start_steps,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/pendulum_td3_actor.pt")
torch.save(critic1.state_dict(), "saves/pendulum_td3_critic1.pt")
torch.save(critic2.state_dict(), "saves/pendulum_td3_critic2.pt")

env.close()

env = gym.make("Pendulum-v1", render_mode="human")


def test(model: torch.nn.Module):
    state, _ = env.reset()
    done = False
    model.eval()

    while not done:
        action = trainer.policy(model, state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()


test(actor)
env.close()
