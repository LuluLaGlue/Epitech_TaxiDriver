import gym
import numpy as np
from IPython.display import clear_output
from time import sleep
import random
import time
import json
import argparse
import pickle


def play(policy, slow=False, render=False, is_loop=False):
    env = gym.make("Taxi-v3")
    env.reset()
    episode = []
    finished = False
    steps = 1
    result = 0

    if render:
        print("Initial Environnement:")
        env.render()

    while not finished:
        s = env.env.s

        timestep = []
        timestep.append(s)
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
            top_range += prob[1]
            if n < top_range:
                action = prob[0]
                break
        _, reward, finished, _ = env.step(action)

        result += reward
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

        if render or (random.uniform(0, 1) < 0.3 and not is_loop):
            print()
            env.render()
        if slow:
            input("Press anything to continue...")
            print("\r", end="\r")
        steps += 1

        if steps >= 100:
            break
    if not is_loop:
        print("[{} MOVES] - Total reward: {}".format(steps, result))

    return steps, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Taxi Driver Game Using the Q-Learning Algorithm"
    )
    parser.add_argument(
        "-s",
        "--slow",
        dest="slow",
        action="store_true",
        default=False,
        help="Activate Slow Mode",
    )
    parser.add_argument("-r",
                        "--render",
                        dest="render",
                        action="store_true",
                        default=False,
                        help="Render State")
    parser.add_argument("-l",
                        "--loop",
                        type=int,
                        help="How many times to play the game",
                        default=1)

    args = parser.parse_args()

    with open('policy.pkl', 'rb') as f:
        policy = pickle.load(f)

    start = time.time()
    mean_steps, mean_result = 0, 0
    total_failed = 0
    is_loop = True if args.loop != 1 else False

    for l in range(args.loop):
        steps, result = play(policy,
                             slow=args.slow,
                             render=args.render,
                             is_loop=is_loop)
        mean_steps += steps
        mean_result += result
        if steps >= 100:
            total_failed += 1

    if is_loop:
        print()
        print(
            "[{} LOOP DONE - {}% FAILED - {} SECONDES] - Mean Steps Per Loop: {} - Mean Reward Per Loop: {}"
            .format(args.loop, np.round(total_failed / args.loop * 100, 2),
                    np.round(time.time() - start, 4),
                    np.round(mean_steps / args.loop, 2),
                    np.round(mean_result / args.loop, 2)))