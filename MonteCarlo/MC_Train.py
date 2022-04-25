import gym
import numpy as np
import random
import time
import argparse
import pickle


def create_random_policy(env) -> dict:
    '''Create an empty dictionary to store state action values'''
    policy = {}
    for key in range(0, env.observation_space.n):
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
        policy[key] = p
    return policy


def create_state_action_dictionary(env, policy: dict) -> dict:
    '''Create an empty dictionary for storing rewards for each state-action pair'''
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q


def run_game(env, policy: dict) -> list[list[int]]:
    '''Plays the game and returns a list of [state, action, reward]'''
    env.reset()
    episode = []
    finished = False

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
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    return episode


def monte_carlo_e_soft(env,
                       episodes: int = 100,
                       epsilon: float = 0.01) -> dict:
    policy = create_random_policy(env)
    Q = create_state_action_dictionary(env, policy)
    returns = {}
    start_episode = time.time()

    for e in range(episodes):
        G = 0  # Store cumulative reward in G (initialized at 0)
        episode = run_game(env=env, policy=policy)

        # for loop through reversed indices of episode array.
        # The logic behind it being reversed is that the eventual reward would be at the end.
        # So we have to go back from the last timestep to the first one propagating result from the future.

        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t  # Increment total reward by reward on current timestep

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:  #
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                Q[s_t][a_t] = sum(returns[state_action]) / len(
                    returns[state_action])  # Average reward across episodes

                Q_list = list(map(
                    lambda x: x[1],
                    Q[s_t].items()))  # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)

                A_star = max_Q

                for a in policy[s_t].items(
                ):  # Update action probability for s_t in policy
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (
                            epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon /
                                             abs(sum(policy[s_t].values())))

        if e % int(episodes / 10) == 0:
            episode_time = (time.time() - start_episode)
            start_episode = time.time()
            print("[EPISODE {}/{}] - {} secondes.".format(
                e,
                episodes,
                np.round(episode_time, 4),
            ))

    with open('policy.pkl', 'wb') as f:
        pickle.dump(policy, f)

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Q-Learning Algorithm")
    parser.add_argument(
        "--episodes",
        type=int,
        default=25000,
        help="Number of episodes",
    )
    args = parser.parse_args()

    env = gym.make("Taxi-v3")
    policy = monte_carlo_e_soft(env, episodes=args.episodes)