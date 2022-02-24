import numpy as np
import argparse
import gym


def play(slow=False, render=False):
    env = gym.make("Taxi-v3")

    q_table = np.load("q-table.npy")
    done = False
    result = 0
    state = env.reset()

    if render:
        print("Initial Environnement:")
        env.render()
    steps = 1

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)

        result += reward
        state = next_state

        if render:
            print()
            env.render()
        steps += 1

        if steps >= 100:
            break

        if slow:
            input("Press anything to continue...")
            print("\r", end="\r")

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
    mean_steps, mean_result = 0, 0
    total_failed = 0

    for l in range(args.loop):
        steps, result = play(slow=args.slow, render=args.render)
        mean_steps += steps
        mean_result += result
        if steps >= 100:
            total_failed += 1

    if args.loop > 1:
        print(
            "[{} LOOP DONE - {}% FAILED] - Mean Steps Per Loop: {} - Mean Reward Per Loop: {}"
            .format(args.loop, np.round(total_failed / args.loop * 100, 2),
                    np.round(mean_steps / args.loop, 2),
                    np.round(mean_result / args.loop, 2)))
