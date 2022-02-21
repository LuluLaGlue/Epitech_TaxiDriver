import gym
import math
import time
import random
import matplotlib
import numpy as np
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


def adjust_learning_rate(episode):
    delta = LR - LR_MIN
    base = LR_MIN
    rate = LR_DECAY
    lr = base + delta * np.exp(-episode / rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def select_action(state, epsilon):
    global steps_done
    sample = random.random()
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(torch.tensor([state
                                            ])).max(1)[1].item()  #.view(1, 1)
    else:
        return env.action_space.sample()


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 400
TARGET_UPDATE = 50
MAX_STEPS_PER_EPISODE = 100
WARMUP_EPISODE = 10
SAVE_FREQ = 1000
LR = 0.001
LR_MIN = 0.0001
LR_DECAY = 5000

n_actions = env.action_space.n
n_observation = env.observation_space.n

policy_net = DQN(n_observation, n_actions).to(device)
target_net = DQN(n_observation, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(50000)

steps_done = 0

episode_durations = []
episode_reward = []


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration & Reward')
    plt.ylim(MAX_STEPS_PER_EPISODE * -2.5, MAX_STEPS_PER_EPISODE + 10)
    plt.plot(durations_t.numpy())
    plt.plot(reward_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

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
                         GAMMA) + reward_batch

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted_q_value, expected_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


print("Start Training...")
num_episodes = 10000
has_finished = False
time_save = int(time.time())
eps_threshold = 1

for i_episode in range(num_episodes):
    state = env.reset()
    reward_per_episode = 0
    steps_per_episode = 0
    if i_episode >= WARMUP_EPISODE:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)

    for t in count():
        # Select and perform an action
        action = select_action(state, eps_threshold)
        next_state, reward, done, info = env.step(action)

        memory.push(torch.tensor([state], device=device),
                    torch.tensor([action], device=device, dtype=torch.long),
                    torch.tensor([next_state], device=device),
                    torch.tensor([reward], device=device),
                    torch.tensor([done], device=device, dtype=torch.bool))

        if done:
            has_finished = done

        if i_episode >= WARMUP_EPISODE:
            optimize_model()
            adjust_learning_rate(i_episode - WARMUP_EPISODE + 1)
            done = (t == MAX_STEPS_PER_EPISODE - 1) or done
        else:
            done = (t == 5 * MAX_STEPS_PER_EPISODE - 1) or done

        state = next_state
        reward_per_episode += reward

        if done:
            steps_per_episode = t
            episode_durations.append(t + 1)
            plot_durations()
            break

    episode_reward.append(reward_per_episode)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("[{}/{} EPISODES] - Total Reward {} In {} Steps - Finished: {}".
              format(i_episode, num_episodes, np.mean(episode_reward[-20:]),
                     steps_per_episode, has_finished))
        has_finished = False
    if i_episode % 1000 == 0 and i_episode != 0:
        torch.save(
            {
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "reward_in_episode": reward_per_episode,
                "episode_durations": steps_per_episode,
            }, f"./models/DQN_{time_save}.pt")

print('Complete')
env.close()
plt.ioff()
plt.show()