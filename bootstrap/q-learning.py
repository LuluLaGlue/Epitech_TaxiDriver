import numpy as np
import gym

env = gym.make("FrozenLake-v1")
n_observations = env.observation_space.n
n_actions = env.action_space.n

print("Observation: ", n_observations)
print("Actions: ", n_actions)

Q_table = np.zeros((n_observations, n_actions))
print("Q Table: \n", Q_table)

n_episodes = 10000
max_iter_episode = 100
exploration_proba = 1
exploration_decreasing_decay = 0.001
min_exploration_proba = 0.01
gamma = 0.99
lr = 0.1

rewards_per_episode = []

for e in range(n_episodes):
    current_state = env.reset()
    done = False
    total_episode_reward = 0

    for _ in range(max_iter_episode):
        if np.random.uniform(0, 1) < exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state, :])

        next_state, reward, done, _ = env.step(action)

        Q_table[current_state,
                action] = (1 - lr) * Q_table[current_state, action] + lr * (
                    reward + gamma * max(Q_table[next_state, :]))
        total_episode_reward = total_episode_reward + reward

        if done:
            break
        current_state = next_state
    exploration_proba = max(min_exploration_proba,
                            np.exp(-exploration_decreasing_decay * e))
    rewards_per_episode.append(total_episode_reward)

print("\nQ Table: \n", Q_table)
print("Mean reward per thousand episodes")
for i in range(10):
    print((i + 1) * 1000, ": mean episode rewards: ",
          np.mean(rewards_per_episode[1000 * i:1000 * (i + 1)]))
