import gym
import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

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


class DQN(nn.Module):

    def __init__(self, input, outputs):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(input, 4)
        self.l1 = nn.Linear(4, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, outputs)

    def forward(self, x):
        x = F.relu(self.l1(self.emb(x)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 400
TARGET_UPDATE = 20
MAX_STEPS_PER_EPISODE = 100
SAVE_FREQ = 1000

n_actions = env.action_space.n
n_observation = env.observation_space.n

policy_net = DQN(n_observation, n_actions).to(device)
target_net = DQN(n_observation, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(50000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(torch.tensor([state
                                            ])).max(1)[1].item()  #.view(1, 1)
    else:
        return env.action_space.sample()


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
    plt.plot(durations_t.numpy())
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        # means_r = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        # means_r = torch.cat((torch.zeros(99), means_r))
        # plt.plot(means_r.numpy())

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

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    reward_per_episode = 0
    steps_per_episode = 0

    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, info = env.step(action)

        reward_per_episode += reward

        memory.push(torch.tensor([state], device=device),
                    torch.tensor([action], device=device, dtype=torch.long),
                    torch.tensor([next_state], device=device),
                    torch.tensor([reward], device=device),
                    torch.tensor([done], device=device, dtype=torch.bool))
        # Observe new state
        if done:
            next_state = None

        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            has_finished = done

        if done or t == MAX_STEPS_PER_EPISODE:
            steps_per_episode = t
            episode_durations.append(t + 1)
            plot_durations()
            break
    episode_reward.append(reward_per_episode)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("[{}/{} EPISODES] - Total Reward {} In {} Steps - Finished: {}".
              format(i_episode, num_episodes, reward_per_episode,
                     steps_per_episode, has_finished))
        has_finished = False
    if i_episode % 1000 == 0:
        torch.save(
            {
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "reward_in_episode": reward_per_episode,
                "episode_durations": steps_per_episode,
            }, f"./models/pytorch_{int(time.time())}.pt")

print('Complete')
env.close()
plt.ioff()
plt.show()