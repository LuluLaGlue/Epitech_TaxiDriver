import gym
import time
import torch
from DQN import DQN
from IPython import display

import torch.optim as optim


def get_action_for_state(state):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        predicted = model(torch.tensor([state], device=device))
        action = predicted.max(1)[1]
    return action.item()


def play(verbose: bool = False, sleep: float = 0.2, max_steps: int = 100):
    # Play an episode
    actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

    iteration = 0
    state = env.reset()  # reset environment to a new, random state
    env.render()
    if verbose:
        print(f"Iter: {iteration} - Action: *** - Reward ***")
    time.sleep(sleep)
    done = False

    while not done:
        action = get_action_for_state(state)
        iteration += 1
        state, reward, done, info = env.step(action)
        display.clear_output(wait=True)
        env.render()
        if verbose:
            print(
                f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}"
            )
        time.sleep(sleep)
        if iteration == max_steps:
            print("cannot converge :(")
            break


if __name__ == "__main__":
    PATH = "./models/DQN_1645447309_almost_converge.pt"
    device = torch.device("cpu")
    env = gym.make("Taxi-v3").env

    n_actions = env.action_space.n
    n_observation = env.observation_space.n
    model = DQN(n_observation, n_actions).to(device)
    optimizer = optim.RMSprop(model.parameters())

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    play(sleep=0.1, max_steps=20)
