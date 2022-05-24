import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse
import time
import gym


def moving_average(x, periods=5):
    if len(x) < periods:

        return x

    cumsum = np.cumsum(np.insert(x, 0, 0))
    res = (cumsum[periods:] - cumsum[:-periods]) / periods

    return np.hstack([x[:periods - 1], res])


def plot_durations(episode_durations,
                   reward_in_episode,
                   epsilon_vec,
                   max_steps_per_episode=100):
    '''Plot graphs containing Epsilon, Rewards, and Steps per episode over time'''
    lines = []
    fig = plt.figure(1, figsize=(15, 7))
    plt.clf()
    ax1 = fig.add_subplot(111)

    plt.title(f'Training...')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration & Rewards')
    ax1.set_ylim(-2 * max_steps_per_episode, max_steps_per_episode + 10)
    ax1.plot(episode_durations, color="C1", alpha=0.2)
    ax1.plot(reward_in_episode, color="C2", alpha=0.2)
    mean_steps = moving_average(episode_durations, periods=5)
    mean_reward = moving_average(reward_in_episode, periods=5)
    lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
    lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon')
    lines.append(ax2.plot(epsilon_vec, label="epsilon", color="C3")[0])
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc=3)

    plt.pause(0.001)


def train(episodes=2000,
          gamma=0.95,
          epsilon=1,
          max_epsilon=1,
          min_epsilon=0.001,
          epsilon_decay=0.01,
          alpha=0.85):
    start_date = datetime.now()
    start_time = time.time()
    total_reward = []
    steps_per_episode = []
    epsilon_vec = []

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    reward = 0

    print("{} - Starting Training...\n".format(start_date))
    start_episode = time.time()

    for e in range(episodes):
        done = False

        total_reward.append(0)
        steps_per_episode.append(0)
        state1 = env.reset()

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * e)
        epsilon_vec.append(epsilon)

        if np.random.uniform(0, 1) < epsilon:
            action1 = env.action_space.sample()
        else:
            action1 = np.argmax(Q[state1, :])

        while not done:
            state2, reward, done, _ = env.step(action1)
            total_reward[e] += reward
            steps_per_episode[e] += 1
            if np.random.uniform(0, 1) < epsilon:
                action2 = env.action_space.sample()
            else:
                action2 = np.argmax(Q[state2, :])

            predict = Q[state1, action1]
            target = reward + gamma * Q[state2, action2]
            Q[state1,
              action1] = Q[state1, action1] + alpha * (target - predict)

            state1 = state2
            action1 = action2

            reward += 1
        if e % int(episodes / 100) == 0:
            episode_time = (time.time() - start_episode)
            print(
                "[EPISODE {}/{}] - Mean reward for last {} Episodes: {} in {} steps - Mean Time Per Episode: {}"
                .format(e, episodes, int(episodes / 100),
                        np.mean(total_reward[-int(episodes / 100):]),
                        np.mean(steps_per_episode[-int(episodes / 100):]),
                        np.round(episode_time / e, 6) if e != 0 else 0))

    end_date = datetime.now()
    execution_time = (time.time() - start_time)
    plot_durations(steps_per_episode,
                   total_reward,
                   epsilon_vec,
                   max_steps_per_episode=200)

    print()
    print("{} - Training Ended".format(end_date))
    print("Mean Reward: {}".format(np.mean(total_reward)))
    print("Time to train: \n    - {}s\n    - {}min\n    - {}h".format(
        np.round(execution_time, 2), np.round(execution_time / 60, 2),
        np.round(execution_time / 3600, 2)))

    np.save("qtable", Q)
    plt.savefig("SARSA_graph.png")

    return np.round(execution_time, 2), np.mean(total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Q-Learning Algorithm")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of episodes",
    )
    parser.add_argument("-a",
                        "--alpha",
                        type=float,
                        default=0.85,
                        help="Alpha Factor")
    parser.add_argument("-g",
                        "--gamma",
                        type=float,
                        default=0.99,
                        help="Discount Rating")
    parser.add_argument("-e",
                        "--epsilon",
                        type=float,
                        default=1,
                        help="Exploration Rate")
    parser.add_argument("--min_epsilon",
                        type=float,
                        default=0.001,
                        help="Minimal value for Exploration Rate")
    parser.add_argument("-d",
                        "--decay_rate",
                        type=float,
                        default=0.01,
                        help="Exponential decay rate for Exploration Rate")

    args = parser.parse_args()

    epsilon = args.epsilon
    max_epsilon = args.epsilon
    episodes = args.episodes
    gamma = args.gamma
    min_epsilon = args.min_epsilon
    epsilon_decay = args.decay_rate
    alpha = args.alpha

    env = gym.make("Taxi-v3")

    train(episodes, gamma, epsilon, max_epsilon, min_epsilon, epsilon_decay,
          alpha)
