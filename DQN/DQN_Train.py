import os
import gym
import time
import argparse
import numpy as np
import pandas as pd
from DQN import DQN
from DQN_Play import play
from Memory import Transition, ReplayMemory
from itertools import count
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.optim as optim
import torch.nn.functional as F

plt.ion()


class TrainingAgent():

    def __init__(self,
                 env=gym.make("Taxi-v3").env,
                 batch_size=128,
                 gamma=0.99,
                 eps_start=1,
                 eps_end=0.1,
                 eps_decay=400,
                 target_update=20,
                 max_steps_per_episode=100,
                 warmup_episode=10,
                 save_freq=1000,
                 lr=0.001,
                 lr_min=0.0001,
                 lr_decay=5000,
                 memory_size=50000,
                 num_episodes=10000):
        self.config = {
            "BATCH_SIZE": batch_size,
            "GAMMA": gamma,
            "EPS_START": eps_start,
            "EPS_END": eps_end,
            "EPS_DECAY": eps_decay,
            "TARGET_UPDATE": target_update,
            "MAX_STEPS_PER_EPISODE": max_steps_per_episode,
            "WARMUP_EPISODE": warmup_episode,
            "SAVE_FREQ": save_freq,
            "LR": lr,
            "LR_MIN": lr_min,
            "LR_DECAY": lr_decay,
            "MEMORY_SIZE": memory_size,
            "NUM_EPISODES": num_episodes
        }
        self.rng = np.random.default_rng(42)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.id = int(time.time())

    def compile(self):
        n_actions = self.env.action_space.n
        n_observations = self.env.observation_space.n

        self.model = DQN(n_observations, n_actions).to(self.device)
        self.target_model = DQN(n_observations, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config["LR"])

    def _get_epsilon(self, episode):
        epsilon = self.config["EPS_END"] + \
                          (self.config["EPS_START"] - self.config["EPS_END"]) * \
                              np.exp(-episode / self.config["EPS_DECAY"])
        return epsilon

    def _get_action_for_state(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            predicted = self.model(torch.tensor([state], device=self.device))
            action = predicted.max(1)[1]
        return action.item()

    def _choose_action(self, state, epsilon):
        if self.rng.uniform() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self._get_action_for_state(state)
        return action

    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(
            torch.tensor([state], device=self.device),
            torch.tensor([action], device=self.device, dtype=torch.long),
            torch.tensor([next_state], device=self.device),
            torch.tensor([reward], device=self.device),
            torch.tensor([done], device=self.device, dtype=torch.bool))

    def _train_model(self):
        if len(self.memory) < self.config["BATCH_SIZE"]:
            return
        transitions = self.memory.sample(self.config["BATCH_SIZE"])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute predicted Q values
        predicted_q_value = self.model(state_batch).gather(
            1, action_batch.unsqueeze(1))

        # Compute the expected Q values
        next_state_values = self.target_model(next_state_batch).max(1)[0]
        expected_q_values = (~done_batch * next_state_values *
                             self.config["GAMMA"]) + reward_batch

        loss = self.loss(predicted_q_value, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _adjust_learning_rate(self, episode):
        delta = self.config["LR"] - self.config["LR_MIN"]
        base = self.config["LR_MIN"]
        rate = self.config["LR_DECAY"]
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def fit(self):
        self.memory = ReplayMemory(self.config["MEMORY_SIZE"])
        self.loss = F.smooth_l1_loss

        self.episode_durations = []
        self.reward_in_episode = []
        self.epsilon_vec = []
        reward_in_episode = 0
        epsilon = 1

        for i_episode in range(self.config["NUM_EPISODES"]):
            state = self.env.reset()
            if i_episode >= self.config["WARMUP_EPISODE"]:
                epsilon = self._get_epsilon(i_episode -
                                            self.config["WARMUP_EPISODE"])
            for step in count():
                action = self._choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self._remember(state, action, next_state, reward, done)

                if i_episode >= self.config["WARMUP_EPISODE"]:
                    self._train_model()
                    self._adjust_learning_rate(i_episode -
                                               self.config["WARMUP_EPISODE"] +
                                               1)
                    done = (step == self.config["MAX_STEPS_PER_EPISODE"] -
                            1) or done
                else:
                    done = (step == 5 * self.config["MAX_STEPS_PER_EPISODE"] -
                            1) or done

                state = next_state
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(step + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    self.epsilon_vec.append(epsilon)
                    reward_in_episode = 0
                    self.plot_durations()
                    break
            if i_episode % self.config["TARGET_UPDATE"] == 0:
                self._update_target()
            if i_episode % self.config["SAVE_FREQ"] == 0:
                self.save()
            self.last_episode = i_episode

    @staticmethod
    def _moving_average(x, periods=5):
        if len(x) < periods:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0))
        res = (cumsum[periods:] - cumsum[:-periods]) / periods
        return np.hstack([x[:periods - 1], res])

    def save(self):
        if not os.path.isdir(f"./models/{self.id}"):
            os.makedirs(f"./models/{self.id}")
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                "reward_in_episode": self.reward_in_episode,
                "episode_durations": self.episode_durations,
                "epsilon_vec": self.epsilon_vec,
                "config": self.config
            }, f"./models/{self.id}/DQN_{self.id}.pt")
        plt.savefig(f"./models/{self.id}/DQN_{self.id}_graph.png")

    def plot_durations(self):
        lines = []
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        ax1 = fig.add_subplot(111)

        plt.title('Training...')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration & Rewards')
        ax1.set_ylim(-2 * self.config["MAX_STEPS_PER_EPISODE"],
                     self.config["MAX_STEPS_PER_EPISODE"] + 10)
        ax1.plot(self.episode_durations, color="C1", alpha=0.2)
        ax1.plot(self.reward_in_episode, color="C2", alpha=0.2)
        mean_steps = self._moving_average(self.episode_durations, periods=5)
        mean_reward = self._moving_average(self.reward_in_episode, periods=5)
        lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
        lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(
            ax2.plot(self.epsilon_vec, label="epsilon", color="C3")[0])
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=3)

        plt.show()
        plt.pause(0.001)


# def adjust_learning_rate(episode, lr, lr_min, lr_decay, optimizer):
#     delta = lr - lr_min
#     base = lr_min
#     rate = lr_decay
#     lr = base + delta * np.exp(-episode / rate)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     return lr, optimizer

# def select_action(state, epsilon, steps_done):
#     # global steps_done
#     sample = random.random()
#     steps_done += 1
#     if sample > epsilon:
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(torch.tensor(
#                 [state])).max(1)[1].item(), steps_done  #.view(1, 1)
#     else:
#         return env.action_space.sample(), steps_done

# def plot_durations(episode_durations, episode_reward, max_steps_per_episode):
#     plt.figure(1)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     reward_t = torch.tensor(episode_reward, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration & Reward')
#     plt.ylim(max_steps_per_episode * -2, max_steps_per_episode + 10)
#     plt.plot(durations_t.numpy())
#     plt.plot(reward_t.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())

# def optimize_model(batch_size, policy_net, target_net, optimizer):
#     if len(memory) < batch_size:
#         return
#     transitions = memory.sample(batch_size)

#     batch = Transition(*zip(*transitions))

#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#     next_state_batch = torch.cat(batch.next_state)
#     done_batch = torch.cat(batch.done)

#     # Compute predicted Q values
#     predicted_q_value = policy_net(state_batch).gather(
#         1, action_batch.unsqueeze(1))

#     # Compute the expected Q values
#     next_state_values = target_net(next_state_batch).max(1)[0]
#     expected_q_values = (~done_batch * next_state_values *
#                          gamma) + reward_batch

#     # Compute loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(predicted_q_value, expected_q_values.unsqueeze(1))

#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()

#     return policy_net, target_net, optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Taxi Driver Model based on DQN")
    parser.add_argument("--environment",
                        type=str,
                        default="Taxi-v3",
                        help="Environment to train the DQN")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch Size")
    parser.add_argument("--gamma", type=float, default=0.99, help="GAMMA")
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
                        default=20,
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
    env = gym.make(args.environment).env
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

    start_time = time.time()

    agent = TrainingAgent(env=env,
                          batch_size=batch_size,
                          gamma=gamma,
                          eps_start=eps_start,
                          eps_end=eps_end,
                          eps_decay=eps_decay,
                          target_update=target_update,
                          max_steps_per_episode=max_steps_per_episode,
                          warmup_episode=warmup_episode,
                          save_freq=save_freq,
                          lr=lr,
                          lr_min=lr_min,
                          lr_decay=lr_decay,
                          memory_size=memory_size,
                          num_episodes=num_episodes)
    agent.compile()

    agent.fit()

    time_train = time.time() - start_time()

    print('Training Complete in {}s - {}min - {}h'.format(
        time_train, np.round(time_train / 60, 2),
        np.round(time_train / 3600, 2)))
    print()
    print("Testing...")

    mean_steps, mean_result, total_failed = 0, 0, 0
    for l in range(1000):
        steps, result, done = play(agent.model)
        mean_steps += steps
        mean_result += result
        if not done:
            total_failed += 1
    percentage_success = np.round((1 - total_failed / 1000) * 100, 2)

    df = pd.read_csv("models.csv", sep=";")

    new_row = [[
        f"{agent.id}/DQN_{agent.id}.pt", agent.config["BATCH_SIZE"],
        agent.config["GAMMA"], agent.config["EPS_START"],
        agent.config["EPS_END"], agent.config["EPS_DECAY"],
        agent.config["TARGET_UPDATE"], agent.config["MAX_STEPS_PER_EPISODE"],
        agent.config["WARMUP_EPISODE"], agent.config["SAVE_FREQ"],
        agent.config["LR"], agent.config["LR_MIN"], agent.config["LR_DECAY"],
        agent.config["MEMORY_SIZE"], agent.config["NUM_EPISODES"],
        np.round(np.mean(agent.reward_in_episode[-100:]),
                 2), percentage_success, time_train
    ]]
    df2 = pd.DataFrame(new_row, columns=df.columns.values)
    new_df = pd.concat([df, df2])
    new_df.set_index('name', drop=True, inplace=True)

    new_df.to_csv("models.csv", sep=";")

    print("Testing Complete.")
