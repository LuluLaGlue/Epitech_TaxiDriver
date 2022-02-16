from datetime import datetime
import numpy as np
import argparse
import random
import time
import gym


def train(episodes=25000,
          lr=0.01,
          gamma=0.99,
          epsilon=1,
          max_epsilon=1,
          min_epsilon=0.001,
          epsilon_decay=0.01):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    total_reward = []
    steps_per_episode = []

    print("{} - Starting Training...\n".format(start_date))
    for e in range(episodes):
        start_episode = time.time()
        state = env.reset()

        done = False
        total_reward.append(0)
        steps_per_episode.append(0)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * e)

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

    end_date = datetime.now()
    execution_time = (time.time() - start_time)

    print()
    print("{} - Training Ended".format(end_date))
    print("Mean Reward: {}".format(np.mean(total_reward)))
    print("Time to train: \n    - {}s\n    - {}min\n    - {}h".format(
        np.round(execution_time, 2), np.round(execution_time / 60, 2),
        np.round(execution_time / 3600, 2)))
    # print("Q Table: \n", q_table)
    np.save("q-table", q_table)

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

    args = parser.parse_args()

    epsilon = args.epsilon
    max_epsilon = args.epsilon
    episodes = args.episodes
    lr = args.learning_rate
    gamma = args.gamma
    min_epsilon = args.min_epsilon
    epsilon_decay = args.decay_rate

    start_date = datetime.now()
    start_time = time.time()

    env = gym.make("Taxi-v3")

    time, reward = train(episodes, lr, gamma, epsilon, max_epsilon,
                         min_epsilon, epsilon_decay)
