# Run with `python3 -m examples.train_snake_dqn`
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from src.envs import SnakeGameEnvironment
from src.trainers import DQNTrainer
from src.buffers import ReplayMemory

env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000
)

trainer = DQNTrainer(env) # Without memory
# trainer = DQNTrainer(env, memory=ReplayMemory(memory_size=256, batch_size=8)) # With memory (uncomment this line)

n_episodes = 300
learning_rate = 0.005#0.005
gamma = 0.9  # Discount factor
min_epsilon = 0.01
max_epsilon = 1
decay_rate = 0.5

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = Model(11, 128, 3)

optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = MSELoss()


trainer.train(
    model = model,
    optimizer = optimizer,
    criterion = criterion,
    n_episodes = n_episodes,
    max_epsilon = max_epsilon,
    min_epsilon = min_epsilon,
    decay_rate = decay_rate,
    decay_epsilon = "linear",
    gamma = gamma,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(model.state_dict(), "saves/snake_dqn.pt")


  

def test(model: torch.nn.Module):
    env.speed = 10
    state, _ = env.reset()
    terminated = False
    while not terminated:
        action = trainer.greedy_policy(model, state)
        state, reward, terminated, _, _ = env.step(action)

test(model)