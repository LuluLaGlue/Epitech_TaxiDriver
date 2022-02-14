import gym

MAX_ITERATION = 10

env = gym.make("FrozenLake-v1")
env.reset()
env.render()

print("Actions Space: ", env.action_space)
print("Observation Space: ", env.observation_space)

for i in range(MAX_ITERATION):
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    print("New State: ", new_state)
    print("Reward: ", reward)
    print("Done: ", done)
    print("Info: ", info)
    env.render()
    if done:
        break
