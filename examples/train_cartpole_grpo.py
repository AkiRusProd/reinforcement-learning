# Run with `python3 -m examples.train_cartpole_grpo`
import torch
import torch.nn as nn
from torch.optim import Adam
from src.trainers import GRPOTrainer
import gymnasium as gym


env = gym.make("CartPole-v1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = GRPOTrainer(env, device=device)

n_episodes = 500
learning_rate = 3e-4
group_size = 8
batch_size = 1024
mini_batch_size = 256
grpo_epochs = 4
clip_param = 0.2
beta = 0.01
entropy_coeff = 0.01
reference_update_interval = 10


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = Adam(actor.parameters(), lr=learning_rate)

trainer.train(
    actor=actor,
    optimizer=optimizer,
    n_episodes=n_episodes,
    group_size=group_size,
    batch_size=batch_size,
    mini_batch_size=mini_batch_size,
    grpo_epochs=grpo_epochs,
    clip_param=clip_param,
    beta=beta,
    entropy_coeff=entropy_coeff,
    reference_update_interval=reference_update_interval,
    max_steps=500,
)

env.close()

env = gym.make("CartPole-v1", render_mode="human")


def test(model: torch.nn.Module):
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            logits = model(torch.tensor(state, dtype=torch.float32).to(device))
            action = torch.argmax(logits)

        state, _, terminated, truncated, _ = env.step(action.item())
        env.render()


test(actor)

env.close()
