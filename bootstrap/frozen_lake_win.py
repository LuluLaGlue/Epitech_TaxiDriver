import gym

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
env.render()

actions = {"Left": 0, "Down": 1, "Right": 2, "Up": 3}
winning_sequence = (2 * ['Right']) + (3 * ['Down']) + ['Right']
for move in winning_sequence:
    new_state, reward, done, info = env.step(actions[move])
    env.render()
    print("Reward: {:.2f}".format(reward))
    print(info)
    if done:
        break