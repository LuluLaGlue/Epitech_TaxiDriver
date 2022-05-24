import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib
import argparse
import random
import time
import gym

is_notebook = 'inline' in matplotlib.get_backend()

if is_notebook:
    from IPython import display


def moving_average(x: list, periods: int = 5) -> list:
    if len(x) < periods:

        return x

    cumsum = np.cumsum(np.insert(x, 0, 0))
    res = (cumsum[periods:] - cumsum[:-periods]) / periods

    return np.hstack([x[:periods - 1], res])


def plot_durations(episode_durations: list,
                   reward_in_episode: list,
                   epsilon_vec: list,
                   max_steps_per_episode: int = 100) -> None:
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

    if is_notebook:
        display.clear_output(wait=True)
    else:
        plt.show()
    plt.pause(0.001)

    return


def train(episodes: int = 25000,
          lr: float = 0.01,
          gamma: float = 0.99,
          epsilon: float = 1,
          max_epsilon: float = 1,
          min_epsilon: float = 0.001,
          epsilon_decay: float = 0.01,
          show_empty: bool = True) -> tuple[float, int]:
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    total_reward = []
    steps_per_episode = []
    epsilon_vec = []

    print("{} - Starting Training...\n".format(start_date))
    for e in range(episodes):
        start_episode = time.time()
        state = env.reset()

        done = False
        total_reward.append(0)
        steps_per_episode.append(0)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * e)
        epsilon_vec.append(epsilon)

        # Loop as long as the game is not over, i.e. done is not True
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore the action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            # Apply the action and see what happens
            next_state, reward, done, _ = env.step(action)
            total_reward[e] += reward
            steps_per_episode[e] += 1

            current_value = q_table[
                state, action]  # current Q-value for the state/action couple
            next_max = np.max(q_table[next_state])  # next best Q-value

            # Compute the new Q-value with the Bellman equation
            q_table[state, action] = (1 - lr) * current_value + lr * (
                reward + gamma * next_max)
            state = next_state

        if e % int(episodes / 100) == 0:
            episode_time = (time.time() - start_episode)
            print(
                "[EPISODE {}/{}] - {}min - Mean reward for last {} Episodes: {} in {} steps"
                .format(e, episodes, np.round(episode_time / 60, 2),
                        int(episodes / 100),
                        np.mean(total_reward[-int(episodes / 100):]),
                        np.mean(steps_per_episode[-int(episodes / 100):])))

    plot_durations(steps_per_episode,
                   total_reward,
                   epsilon_vec,
                   max_steps_per_episode=200)
    end_date = datetime.now()
    execution_time = (time.time() - start_time)

    print()
    print("{} - Training Ended".format(end_date))
    print("Mean Reward: {}".format(np.mean(total_reward)))
    print("Time to train: \n    - {}s\n    - {}min\n    - {}h".format(
        np.round(execution_time, 2), np.round(execution_time / 60, 2),
        np.round(execution_time / 3600, 2)))

    if show_empty:
        total_empty = 0
        for i, q in enumerate(q_table):
            if 0 in q:
                total_empty += 1
        print("Found {} empty lines in the Q Table - {}%.".format(
            total_empty, int((total_empty / len(q_table) * 100))))

    np.save("qtable", q_table)
    plt.show()
    plt.savefig("Q-Learning_graph.png")

    return np.round(execution_time, 2), np.mean(total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Q-Learning Algorithm")
    parser.add_argument(
        "--episodes",
        type=int,
        default=25000,
        help="Number of episodes",
    )
    parser.add_argument("-l",
                        "--learning_rate",
                        type=float,
                        default=0.01,
                        help="Learning Rate")
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
    parser.add_argument("--show_empty",
                        dest="empty",
                        action="store_true",
                        default=True,
                        help="Render State")

    args = parser.parse_args()

    plt.ion()

    epsilon = args.epsilon
    max_epsilon = args.epsilon
    episodes = args.episodes
    lr = args.learning_rate
    gamma = args.gamma
    min_epsilon = args.min_epsilon
    epsilon_decay = args.decay_rate
    show_empty = args.empty

    start_date = datetime.now()
    start_time = time.time()

    env = gym.make("Taxi-v3")

    time, reward = train(episodes, lr, gamma, epsilon, max_epsilon,
                         min_epsilon, epsilon_decay, show_empty)
    plt.ioff()