import os
import gym
import time
import random
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
            "batch_size": batch_size,
            "gamma": gamma,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "target_update": target_update,
            "max_steps_per_episode": max_steps_per_episode,
            "warmup_episode": warmup_episode,
            "save_freq": save_freq,
            "lr": lr,
            "lr_min": lr_min,
            "lr_decay": lr_decay,
            "memory_size": memory_size,
            "num_episodes": num_episodes
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
                                    lr=self.config["lr"])

    def _get_epsilon(self, episode):
        epsilon = self.config["eps_end"] + \
                          (self.config["eps_start"] - self.config["eps_end"]) * \
                              np.exp(-episode / self.config["eps_decay"])
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
        if len(self.memory) < self.config["batch_size"]:
            return
        transitions = self.memory.sample(self.config["batch_size"])
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
                             self.config["gamma"]) + reward_batch

        loss = self.loss(predicted_q_value, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _adjust_learning_rate(self, episode):
        delta = self.config["lr"] - self.config["lr_min"]
        base = self.config["lr_min"]
        rate = self.config["lr_decay"]
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def fit(self):
        self.memory = ReplayMemory(self.config["memory_size"])
        self.loss = F.smooth_l1_loss

        self.episode_durations = []
        self.reward_in_episode = []
        self.epsilon_vec = []
        reward_in_episode = 0
        epsilon = 1

        for i_episode in range(self.config["num_episodes"]):
            state = self.env.reset()
            if i_episode >= self.config["warmup_episode"]:
                epsilon = self._get_epsilon(i_episode -
                                            self.config["warmup_episode"])
            for step in count():
                action = self._choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self._remember(state, action, next_state, reward, done)

                if i_episode >= self.config["warmup_episode"]:
                    self._train_model()
                    self._adjust_learning_rate(i_episode -
                                               self.config["warmup_episode"] +
                                               1)
                    done = (step == self.config["max_steps_per_episode"] -
                            1) or done
                else:
                    done = (step == 5 * self.config["max_steps_per_episode"] -
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
            if i_episode % self.config["target_update"] == 0:
                self._update_target()
            if i_episode % self.config["save_freq"] == 0:
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
        ax1.set_ylim(-2 * self.config["max_steps_per_episode"],
                     self.config["max_steps_per_episode"] + 10)
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

    # n_actions = env.action_space.n
    # n_observation = env.observation_space.n

    # policy_net = DQN(n_observation, n_actions).to(device)
    # target_net = DQN(n_observation, n_actions).to(device)
    # target_net.load_state_dict(policy_net.state_dict())
    # target_net.eval()

    # optimizer = optim.RMSprop(policy_net.parameters())
    # memory = ReplayMemory(memory_size)

    # steps_done = 0

    # episode_durations = []
    # episode_reward = []

    # print("Start Training...")
    # has_finished = False
    # time_save = int(time.time())
    # model_name = "DQN_{}.pt".format(time_save)
    # eps_threshold = 1

    # for i_episode in range(num_episodes):
    #     state = env.reset()
    #     reward_per_episode = 0
    #     steps_per_episode = 0
    #     if i_episode >= warmup_episode:
    #         eps_threshold = eps_end + (eps_start - eps_end) * \
    #             math.exp(-1. * steps_done / eps_decay)

    #     for t in count():
    #         # Select and perform an action
    #         action, steps_done = select_action(state, eps_threshold,
    #                                            steps_done)
    #         next_state, reward, done, info = env.step(action)

    #         memory.push(
    #             torch.tensor([state], device=device),
    #             torch.tensor([action], device=device, dtype=torch.long),
    #             torch.tensor([next_state], device=device),
    #             torch.tensor([reward], device=device),
    #             torch.tensor([done], device=device, dtype=torch.bool))

    #         if done:
    #             has_finished = done

    #         if i_episode >= warmup_episode:
    #             policy_net, target_net, optimizer = optimize_model(
    #                 batch_size, policy_net, target_net, optimizer)
    #             lr, optimizer = adjust_learning_rate(
    #                 i_episode - warmup_episode + 1, lr, lr_min, lr_decay,
    #                 optimizer)
    #             done = (t == max_steps_per_episode - 1) or done
    #         else:
    #             done = (t == 5 * max_steps_per_episode - 1) or done

    #         state = next_state
    #         reward_per_episode += reward

    #         if done:
    #             steps_per_episode = t
    #             episode_durations.append(t + 1)
    #             plot_durations(episode_durations, episode_reward,
    #                            max_steps_per_episode)
    #             break

    #     episode_reward.append(reward_per_episode)
    #     # Update the target network, copying all weights and biases in DQN
    #     if i_episode % target_update == 0:
    #         target_net.load_state_dict(policy_net.state_dict())
    #         print(
    #             "[{}/{} EPISODES] - Total Reward {} In {} Steps - Finished: {}"
    #             .format(i_episode, num_episodes,
    #                     np.mean(episode_reward[-target_update:]),
    #                     steps_per_episode, has_finished))
    #         has_finished = False
    #     if i_episode % 1000 == 0 and i_episode != 0:
    # if not os.path.isdir(f"./models/{time_save}"):
    #     os.makedirs(f"./models/{time_save}")
    #         torch.save(
    #             {
    #                 'model_state_dict': policy_net.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 "reward_in_episode": reward_per_episode,
    #                 "episode_durations": steps_per_episode,
    #             }, f"./models/{time_save}/DQN_{time_save}.pt")

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
        f"{agent.id}/DQN_{agent.id}.pt", agent.config["batch_size"],
        agent.config["gamma"], agent.config["eps_start"],
        agent.config["eps_end"], agent.config["eps_decay"],
        agent.config["target_update"], agent.config["max_steps_per_episode"],
        agent.config["warmup_episode"], agent.config["save_freq"],
        agent.config["lr"], agent.config["lr_min"], agent.config["lr_decay"],
        agent.config["memory_size"], agent.config["num_episodes"],
        np.round(np.mean(agent.reward_in_episode[-100:]),
                 2), percentage_success
    ]]
    df2 = pd.DataFrame(new_row, columns=df.columns.values)
    new_df = pd.concat([df, df2])
    new_df.set_index('name', drop=True, inplace=True)

    new_df.to_csv("models.csv", sep=";")

    # df = pd.read_csv("models.csv", sep=";")

    # new_row = [[
    #     f"{time_save}/{model_name}", batch_size, gamma, eps_start, eps_end,
    #     eps_decay, target_update, max_steps_per_episode, warmup_episode,
    #     save_freq, lr, lr_min, lr_decay, memory_size, num_episodes,
    #     np.round(np.mean(episode_reward[-100:]), 2), percentage_success
    # ]]
    # df2 = pd.DataFrame(new_row, columns=df.columns.values)
    # new_df = pd.concat([df, df2])
    # new_df.set_index('name', drop=True, inplace=True)

    # new_df.to_csv("models.csv", sep=";")
    # plt.savefig("models/{0}/DQN_{0}_graph.png".format(time_save))

    print('Complete')
    # env.close()
    # plt.ioff()
    # plt.show()