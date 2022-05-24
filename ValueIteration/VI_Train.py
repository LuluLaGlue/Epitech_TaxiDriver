from datetime import datetime
import numpy as np
import argparse
import time
import gym


def choose_action(state: int, n_actions: int, V: dict) -> int:
    best_action = None
    best_value = float('-inf')

    for action in range(0, n_actions):
        env.env.s = state

        new_state, reward, done, info = env.step(action)
        v = reward + gamma * V[new_state]

        if v > best_value:
            best_value = v
            best_action = action

    return best_action


def train(env=gym.make("Taxi-v3"),
          gamma: float = 0.9,
          significant_improvement: float = 0.001) -> tuple[float, int]:
    env.reset()
    n_actions = env.action_space.n
    n_observations = env.observation_space.n
    V = np.zeros([n_observations])
    Pi = np.zeros([n_observations], dtype=int)

    total_reward = []
    steps_per_episode = []
    iteration = 0

    print("{} - Starting Training...\n".format(start_date))
    while True:
        total_reward.append(0)
        steps_per_episode.append(0)
        biggest_change = 0
        for state in range(0, n_observations):
            old_v = V[state]
            action = choose_action(state, n_actions, V)
            env.env.s = state

            new_state, reward, done, info = env.step(action)
            total_reward[iteration] += reward
            steps_per_episode[iteration] += 1

            V[state] = reward + gamma * V[new_state]
            Pi[state] = action
            biggest_change = max(biggest_change, np.abs(old_v - V[state]))
        iteration += 1

        if biggest_change < significant_improvement:
            print("[EPISODE {}] - Mean reward during training: {} in {} steps".
                  format(iteration, np.mean(total_reward),
                         np.mean(steps_per_episode)))
            break
    end_date = datetime.now()
    execution_time = (time.time() - start_time)

    print()
    print("{} - Training Ended".format(end_date))
    print("Mean Reward: {}".format(np.mean(total_reward)))
    print("Time to train: \n    - {}s\n    - {}min\n    - {}h".format(
        np.round(execution_time, 2), np.round(execution_time / 60, 2),
        np.round(execution_time / 3600, 2)))
    np.save("v-iteration", Pi)

    return np.round(execution_time, 2), np.mean(total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Value Iteration Algorithm")
    parser.add_argument(
        "--gamma",
        type=int,
        default=0.9,
        help="Gamma",
    )
    args = parser.parse_args()

    gamma = args.gamma
    start_date = datetime.now()
    start_time = time.time()

    env = gym.make("Taxi-v3")

    train(env=env, gamma=gamma, significant_improvement=0.01)