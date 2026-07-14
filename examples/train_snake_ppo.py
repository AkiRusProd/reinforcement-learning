# Run with `python3 -m examples.train_snake_ppo`
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from src.envs import SnakeGameEnvironment
from src.trainers import PPOTrainer


env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000,
    render_enabled=False,
)


trainer = PPOTrainer(env)

n_episodes = 1000
learning_rate = 0.005
gamma = 0.99
batch_size = 512
mini_batch_size = 64
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
        self.ax.set_title("Snake PPO rewards")
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
        logits = self.fc2(x)
        return logits

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

actor = Actor(11, hidden_size, 3)
critic = Critic(11, hidden_size)

actor_optimizer = Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = Adam(critic.parameters(), lr=learning_rate)
reward_plotter = RewardPlotter(plot_update_interval, moving_average_window)

scores = trainer.train(
    actor=actor,
    critic=critic,
    criterion=nn.MSELoss(),
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    n_episodes=n_episodes,
    batch_size=batch_size,
    mini_batch_size=mini_batch_size,
    ppo_epochs=4,
    gamma=gamma,
    clip_param=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01,
    max_steps=1000,
    on_episode_end=reward_plotter,
    advantage="gae",
    gae_lambda=0.95,
)
reward_plotter.finalize()

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/snake_ppo_actor.pt")
torch.save(critic.state_dict(), "saves/snake_ppo_critic.pt")


def test(actor: torch.nn.Module):
    env.set_render_enabled(True)
    env.speed = 10
    state, _ = env.reset()
    terminated = False
    while not terminated:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = actor(state_tensor)
            action = torch.argmax(logits).item()
        state, reward, terminated, _, _ = env.step(action)

test(actor)
