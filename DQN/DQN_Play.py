import gym
import torch
import argparse
import numpy as np
from DQN import DQN
from IPython import display

import torch.optim as optim


def import_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = env.action_space.n
    n_observation = env.observation_space.n
    model = DQN(n_observation, n_actions).to(device)
    optimizer = optim.RMSprop(model.parameters())

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer, model, device


def get_action_for_state(state, model, device):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        predicted = model(torch.tensor([state], device=device))
        action = predicted.max(1)[1]
    return action.item()


def play(model,
         device=torch.device("cpu"),
         env=gym.make("Taxi-v3").env,
         render: bool = False,
         max_steps: int = 100,
         slow: bool = False,
         silent: bool = False):
    # Play an episode
    actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

    iteration = 0
    state = env.reset()  # reset environment to a new, random state
    if render:
        env.render()
        print(f"Iter: {iteration} - Action: *** - Reward ***")
    done = False
    total_reward = 0

    while not done:
        action = get_action_for_state(state, model, device)
        iteration += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        display.clear_output(wait=True)
        if render:
            env.render()
            print(
                f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}"
            )
        if iteration == max_steps:
            break
        elif slow and not done:
            input("Press anything to continue...")
            print("\r", end="\r")
    if not silent:
        print("[{}/{} MOVES] - Total reward: {}".format(
            iteration, max_steps, total_reward))

    return iteration, total_reward, done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Taxi Driver Game Using a DQN model")
    parser.add_argument("-p",
                        "--path",
                        type=str,
                        default="./models/DQN_1645447309_almost_converge.pt",
                        help="DQN model to use")
    parser.add_argument(
        "-m",
        "--max",
        type=int,
        default=100,
        help=
        "Max Number of Steps the Model is allowed to take to complete the game"
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
                        help="Render State for each Step")
    parser.add_argument("-l",
                        "--loop",
                        type=int,
                        help="How many times to play the game",
                        default=1)
    args = parser.parse_args()
    path = args.path
    render = args.render
    slow = args.slow
    max = args.max
    loop = args.loop

    if slow:
        render = True

    env = gym.make("Taxi-v3").env
    optimizer, model, device = import_model(path)

    mean_steps, mean_result = 0, 0
    total_failed = 0

    for l in range(loop):
        steps, result, done = play(model,
                                   render=render,
                                   slow=slow,
                                   env=env,
                                   max_steps=max,
                                   device=device)
        mean_steps += steps
        mean_result += result
        if not done:
            total_failed += 1

    if loop > 1:
        print()
        print(
            "[{} LOOP DONE - {}% FAILED] - Mean Steps Per Loop: {} - Mean Reward Per Loop: {}"
            .format(loop, np.round(total_failed / loop * 100, 2),
                    np.round(mean_steps / loop, 2),
                    np.round(mean_result / loop, 2)))
