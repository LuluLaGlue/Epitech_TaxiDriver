import numpy as np
import datetime
import argparse
import random
import pickle
import time
import gym
import sys


def play(policy: dict,
         slow: bool = False,
         render: bool = False,
         is_loop: bool = False,
         is_time: bool = False) -> tuple[int, int]:
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

        if render or (random.uniform(0, 1) < 0.3 and not is_loop
                      and not is_time):
            print()
            env.render()
        if slow:
            input("Press anything to continue...")
            print("\r", end="\r")
        steps += 1

        if steps >= 100:
            break

    if not is_loop and not is_time:
        print("[{} MOVES] - Total reward: {}".format(steps, result))

    return steps, result


def display_data(total, total_failed, start, mean_steps, mean_result):
    print()
    print(
        "[{} LOOP DONE - {}% FAILED - {} SECONDES] - Mean Steps Per Loop: {} - Mean Reward Per Loop: {} - Mean Time Per Loop {}"
        .format(total, np.round(total_failed / total * 100, 2),
                np.round(time.time() - start, 4),
                np.round(mean_steps / total, 2),
                np.round(mean_result / total, 2),
                np.round((time.time() - start) / total), 6))


def solve(policy, mean_steps, mean_result, total_failed, slow, render, is_time,
          is_loop):
    steps, result = play(policy,
                         slow=slow,
                         render=render,
                         is_loop=is_loop,
                         is_time=is_time)
    mean_steps += steps
    mean_result += result
    if steps >= 100:
        total_failed += 1

    return mean_steps, mean_result, total_failed


def error_args(args):
    time = args.time
    loop = args.loop

    if time < 0:
        return 1, "Time can not be negative or null."
    if loop <= 0:
        return 1, "Number of loop can not be negative or null"

    return 0, ""


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
    parser.add_argument("-t",
                        "--time",
                        type=int,
                        default=0,
                        help="Run play for x seconds")

    args = parser.parse_args()

    code, msg = error_args(args)

    if code != 0:
        print("[ERROR] - {}".format(msg))

        sys.exit(1)

    with open('policy.pkl', 'rb') as f:
        policy = pickle.load(f)

    start = time.time()
    mean_steps, mean_result = 0, 0
    total_failed = 0
    is_loop = True if args.loop != 1 else False
    maxrt = datetime.timedelta(seconds=args.time) if args.time != 0 else None

    if maxrt != None:
        stop = datetime.datetime.now() + maxrt
        total = 0

        while datetime.datetime.now() < stop:
            mean_steps, mean_result, total_failed = solve(policy,
                                                          mean_steps,
                                                          mean_result,
                                                          total_failed,
                                                          slow=args.slow,
                                                          render=args.render,
                                                          is_time=True,
                                                          is_loop=is_loop)
            total += 1
        display_data(total, total_failed, start, mean_steps, mean_result)
    else:
        for l in range(args.loop):
            mean_steps, mean_result, total_failed = solve(policy,
                                                          mean_steps,
                                                          mean_result,
                                                          total_failed,
                                                          slow=args.slow,
                                                          render=args.render,
                                                          is_time=False,
                                                          is_loop=is_loop)

        if is_loop:
            display_data(args.loop, total_failed, start, mean_steps,
                         mean_result)
