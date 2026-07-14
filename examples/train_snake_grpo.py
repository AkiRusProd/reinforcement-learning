# Run with `python3 -m examples.train_snake_grpo`
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from src.envs import SnakeGameEnvironment
from src.trainers import GRPOTrainer


env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000,
    render_enabled=False,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = GRPOTrainer(env, device=device)

n_episodes = 1000
learning_rate = 0.005
group_size = 8
batch_size = 1024
mini_batch_size = 256
grpo_epochs = 4
clip_param = 0.2
beta = 0.001
entropy_coeff = 0.01
reference_update_interval = 10
hidden_size = 128
plot_update_interval = 1
moving_average_window = 50


class RewardPlotter:
    def __init__(self, update_interval=1, moving_average_window=50):
        self.update_interval = update_interval
        self.moving_average_window = moving_average_window

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.reward_line, = self.ax.plot([], [], label="Episode reward", alpha=0.35)
        self.average_line, = self.ax.plot([], [], label=f"{moving_average_window}-episode average")
        self.ax.set_title("Snake GRPO rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend()

    def __call__(self, episode, score, scores):
        if (episode + 1) % self.update_interval != 0 and episode != n_episodes - 1:
            return

        episodes = list(range(1, len(scores) + 1))
        moving_average = self._moving_average(scores)

        self.reward_line.set_data(episodes, scores)
        self.average_line.set_data(episodes, moving_average)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def finalize(self):
        plt.ioff()
        self.fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)

    def _moving_average(self, values):
        averages = []
        running_sum = 0
        for index, value in enumerate(values):
            running_sum += value
            if index >= self.moving_average_window:
                running_sum -= values[index - self.moving_average_window]
            count = min(index + 1, self.moving_average_window)
            averages.append(running_sum / count)
        return averages


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


actor = Actor(11, hidden_size, 3).to(device)
optimizer = Adam(actor.parameters(), lr=learning_rate)
reward_plotter = RewardPlotter(plot_update_interval, moving_average_window)

scores = trainer.train(
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
    max_steps=1000,
    on_episode_end=reward_plotter,
)
reward_plotter.finalize()

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/snake_grpo_actor.pt")


def test(actor: torch.nn.Module):
    env.set_render_enabled(True)
    env.speed = 10
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = actor(state_tensor)
            action = torch.argmax(logits).item()
        state, _, terminated, truncated, _ = env.step(action)


test(actor)
