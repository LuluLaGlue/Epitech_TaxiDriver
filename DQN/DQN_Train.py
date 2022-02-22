import os
import gym
import math
import time
import random
import argparse
import matplotlib
import numpy as np
import pandas as pd
from DQN import DQN
from itertools import count
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('Taxi-v3').env

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def adjust_learning_rate(episode, lr, lr_min, lr_decay):
    delta = lr - lr_min
    base = lr_min
    rate = lr_decay
    lr = base + delta * np.exp(-episode / rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def select_action(state, epsilon, steps_done):
    # global steps_done
    sample = random.random()
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(torch.tensor(
                [state])).max(1)[1].item(), steps_done  #.view(1, 1)
    else:
        return env.action_space.sample(), steps_done


def plot_durations(episode_durations, episode_reward, max_steps_per_episode):
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration & Reward')
    plt.ylim(max_steps_per_episode * -2.5, max_steps_per_episode + 10)
    plt.plot(durations_t.numpy())
    plt.plot(reward_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model(batch_size, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    # Compute predicted Q values
    predicted_q_value = policy_net(state_batch).gather(
        1, action_batch.unsqueeze(1))

    # Compute the expected Q values
    next_state_values = target_net(next_state_batch).max(1)[0]
    expected_q_values = (~done_batch * next_state_values *
                         gamma) + reward_batch

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted_q_value, expected_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return policy_net, target_net, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Taxi Driver Model based on DQN")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch Size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma")
    parser.add_argument("--eps_start",
                        type=float,
                        default=1.0,
                        help="Epsilon Starting value")
    parser.add_argument("--eps_end",
                        type=float,
                        default=0.1,
                        help="Epsilon minimal value")
    parser.add_argument("--eps_decay",
                        type=float,
                        default=400,
                        help="Epsilon Decay rate")
    parser.add_argument("--target_update",
                        type=int,
                        default=50,
                        help="Number of episodes between dict saving")
    parser.add_argument("--max_steps",
                        type=int,
                        default=100,
                        help="Max Steps Per Episode")
    parser.add_argument("--warmup_episode",
                        type=int,
                        default=10,
                        help="Number of warmup episodes")
    parser.add_argument("--save_freq",
                        type=int,
                        default=1000,
                        help="Number of episode between model saving")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning Rate")
    parser.add_argument("--lr_min",
                        type=float,
                        default=0.0001,
                        help="Learning Rate Minimal Value")
    parser.add_argument("--lr_decay",
                        type=float,
                        default=5000,
                        help="Learning Rate Decay rate")
    parser.add_argument("--memory",
                        type=int,
                        default=50000,
                        help="Size of Memory")
    parser.add_argument("--episodes",
                        type=int,
                        default=10000,
                        help="Number of episodes during training")

    args = parser.parse_args()
    batch_size = args.batch_size
    gamma = args.gamma
    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_decay = args.eps_decay
    target_update = args.target_update
    max_steps_per_episode = args.max_steps
    warmup_episode = args.warmup_episode
    save_freq = args.save_freq
    lr = args.lr
    lr_min = args.lr_min
    lr_decay = args.lr_decay
    memory_size = args.memory
    num_episodes = args.episodes

    n_actions = env.action_space.n
    n_observation = env.observation_space.n

    policy_net = DQN(n_observation, n_actions).to(device)
    target_net = DQN(n_observation, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(memory_size)

    steps_done = 0

    episode_durations = []
    episode_reward = []

    print("Start Training...")
    has_finished = False
    time_save = int(time.time())
    model_name = "DQN_{}.pt".format(time_save)
    eps_threshold = 1

    for i_episode in range(num_episodes):
        state = env.reset()
        reward_per_episode = 0
        steps_per_episode = 0
        if i_episode >= warmup_episode:
            eps_threshold = eps_end + (eps_start - eps_end) * \
                math.exp(-1. * steps_done / eps_decay)

        for t in count():
            # Select and perform an action
            action, steps_done = select_action(state, eps_threshold,
                                               steps_done)
            next_state, reward, done, info = env.step(action)

            memory.push(
                torch.tensor([state], device=device),
                torch.tensor([action], device=device, dtype=torch.long),
                torch.tensor([next_state], device=device),
                torch.tensor([reward], device=device),
                torch.tensor([done], device=device, dtype=torch.bool))

            if done:
                has_finished = done

            if i_episode >= warmup_episode:
                policy_net, target_net, optimizer = optimize_model(
                    batch_size, policy_net, target_net, optimizer)
                lr = adjust_learning_rate(i_episode - warmup_episode + 1, lr,
                                          lr_min, lr_decay)
                done = (t == max_steps_per_episode - 1) or done
            else:
                done = (t == 5 * max_steps_per_episode - 1) or done

            state = next_state
            reward_per_episode += reward

            if done:
                steps_per_episode = t
                episode_durations.append(t + 1)
                plot_durations(episode_durations, episode_reward,
                               max_steps_per_episode)
                break

        episode_reward.append(reward_per_episode)
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(
                "[{}/{} EPISODES] - Total Reward {} In {} Steps - Finished: {}"
                .format(i_episode, num_episodes, np.mean(episode_reward[-20:]),
                        steps_per_episode, has_finished))
            has_finished = False
        if i_episode % 1000 == 0 and i_episode != 0:
            if not os.path.isdir(f"./models/{time_save}"):
                os.makedirs(f"./models/{time_save}")
            torch.save(
                {
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "reward_in_episode": reward_per_episode,
                    "episode_durations": steps_per_episode,
                }, f"./models/{time_save}/DQN_{time_save}.pt")

    df = pd.read_csv("models.csv", sep=";")

    new_row = [[
        f"{time_save}/{model_name}", batch_size, gamma, eps_start, eps_end,
        eps_decay, target_update, max_steps_per_episode, warmup_episode,
        save_freq, lr, lr_min, lr_decay, memory_size, num_episodes,
        np.round(np.mean(episode_reward), 2)
    ]]
    df2 = pd.DataFrame(new_row, columns=df.columns.values)
    new_df = pd.concat([df, df2])
    new_df.set_index('name', drop=True, inplace=True)

    new_df.to_csv("models.csv", sep=";")
    plt.savefig("models/{0}/DQN_{0}_graph.png".format(time_save))

    print('Complete')
    env.close()
    plt.ioff()
    plt.show()