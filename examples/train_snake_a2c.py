# Run with `python3 -m examples.train_snake_a2c`
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.envs import SnakeGameEnvironment
from src.trainers import A2CTrainer


env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000,
    render_enabled=False,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = A2CTrainer(env, device=device)

n_episodes = 1000
learning_rate = 1e-3
gamma = 0.95
n_steps = 5
hidden_size = 128
max_steps = 500


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


actor = Actor(11, hidden_size, 3).to(device)
critic = Critic(11, hidden_size).to(device)

actor_optimizer = Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = Adam(critic.parameters(), lr=learning_rate)

trainer.train(
    actor=actor,
    critic=critic,
    criterion=nn.MSELoss(),
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    n_episodes=n_episodes,
    gamma=gamma,
    value_coeff=0.5,
    entropy_coeff=0.02,
    max_steps=max_steps,
    n_steps=n_steps,
    advantage="gae",
    gae_lambda=0.95,
    normalize_advantages=True,
    max_grad_norm=0.5,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/snake_a2c_actor.pt")
torch.save(critic.state_dict(), "saves/snake_a2c_critic.pt")


def test(actor: torch.nn.Module):
    env.set_render_enabled(True)
    env.speed = 10
    actor.eval()

    state, _ = env.reset()
    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated) and step < max_steps:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits = actor(state_tensor)
            action = torch.argmax(logits).item()

        state, _, terminated, truncated, _ = env.step(action)
        step += 1


test(actor)
