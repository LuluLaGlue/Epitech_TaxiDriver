import numpy as np
import argparse
import gym


def play(slow=False):
    env = gym.make("Taxi-v3")

    q_table = np.load("q-table.npy")
    done = False
    result = 0
    state = env.reset()
    print("Initial Environnement:")
    env.render()
    steps = 1

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)

        result += reward
        state = next_state
        print()
        env.render()
        steps += 1
        if slow:
            input("Press anything to continue...")
            print("\r", end="\r")

    print("DONE")
    print("[{} MOVES] - Total reward: {}".format(steps, result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Q-Learning Algorithm")
    parser.add_argument(
        "-s",
        "--slow",
        dest="slow",
        action="store_true",
        default=False,
        help="Activate Slow Mode",
    )

    args = parser.parse_args()
    play(slow=args.slow)