import numpy as np
import argparse
import time
import gym


def play(is_loop=False):
    total_steps = 0
    total_reward = 0
    passenger_found = False

    state = env.reset()

    while True:
        stop = False
        while not stop:  #int(state / 100) != 0:
            new_state, reward, _, _ = env.step(1)

            if state == new_state:
                stop = True

            state = new_state
            total_steps += 1
            total_reward += reward

        if not is_loop:
            print("TOP REACHED")
            env.render()

        stop = False
        while not stop:  # int(state / 10) != 0:
            new_state, reward, _, _ = env.step(3)
            total_steps += 1
            total_reward += reward

            if new_state == state:

                for s in [0, 0, 3, 3, 1, 1]:
                    new_state, reward, _, _ = env.step(s)
                    total_reward += reward
                    total_steps += 1

                stop = False
                while not stop:  #int(state / 100) != 0:
                    new_state, reward, _, _ = env.step(1)

                    if state == new_state:
                        stop = True

                    state = new_state
                    total_steps += 1
                    total_reward += reward

            state = new_state

        if not is_loop:
            print("TOP LEFT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1
            state = new_state

            if reward == -1:
                passenger_found = True

                if not is_loop:
                    print("Passenger Found...")

        else:
            if not is_loop:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop:
                    print("Passenger Dropped Off")

                break

        for s in [0, 0, 0, 0]:
            new_state, reward, _, _ = env.step(s)
            total_reward += reward
            total_steps += 1

        if not is_loop:
            print("BOTTOM LEFT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1

            if reward == -1:
                passenger_found = True

                if not is_loop:
                    print("Passenger Found...")

        else:
            if not is_loop:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop:
                    print("Passenger Dropped Off")

                break

        for s in [1, 1, 2, 2, 2, 2, 1, 1]:
            new_state, reward, _, _ = env.step(s)
            total_reward += reward
            total_steps += 1

        if not is_loop:
            print("TOP RIGHT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1

            if reward == -1:
                passenger_found = True

                if not is_loop:
                    print("Passenger Found...")

        else:
            if not is_loop:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop:
                    print("Passenger Dropped Off")

                break

        for s in [0, 0, 0, 0, 3]:
            new_state, reward, _, _ = env.step(s)
            total_reward += reward
            total_steps += 1

        if not is_loop:
            print("BOTTOM RIGHT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1

            if reward == -1:
                passenger_found = True

                if not is_loop:
                    print("Passenger Found...")

        else:
            if not is_loop:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop:
                    print("Passenger Dropped Off")

                break

    if not is_loop:
        print("[DONE] {} STEPS TOTAL / {} REWARD TOTAL".format(
            total_steps, total_reward))

    return total_steps, total_reward


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

    start = time.time()
    mean_steps, mean_result = 0, 0
    max_steps = []
    is_loop = True if args.loop != 1 else False

    env = gym.make("Taxi-v3")

    for l in range(args.loop):
        steps, result = play(is_loop=is_loop)
        max_steps.append(steps)
        mean_steps += steps
        mean_result += result

    if is_loop:
        print()
        print(
            "[{} LOOP DONE -  {} SECONDES] - Mean Steps Per Loop: {} - Max Steps For a Loop: {} - Mean Reward Per Loop: {}"
            .format(args.loop, np.round(time.time() - start, 4),
                    np.round(mean_steps / args.loop, 2), np.max(max_steps),
                    np.round(mean_result / args.loop, 2)))
