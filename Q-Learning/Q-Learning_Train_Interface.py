import gym
import time
import random
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


def moving_average(x: list, periods: int = 5) -> list:
    if len(x) < periods:

        return x

    cumsum = np.cumsum(np.insert(x, 0, 0))
    res = (cumsum[periods:] - cumsum[:-periods]) / periods

    return np.hstack([x[:periods - 1], res])


def plot_durations(episode_durations: list,
                   reward_in_episode: list,
                   epsilon_vec: list,
                   max_steps_per_episode: int = 100) -> None:
    '''Plot graphs containing Epsilon, Rewards, and Steps per episode over time'''
    lines = []
    fig = plt.figure(1, figsize=(15, 7))
    plt.clf()
    ax1 = fig.add_subplot(111)

    plt.title(f'Training...')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration & Rewards')
    ax1.set_ylim(-2 * max_steps_per_episode, max_steps_per_episode + 10)
    ax1.plot(episode_durations, color="C1", alpha=0.2)
    ax1.plot(reward_in_episode, color="C2", alpha=0.2)
    mean_steps = moving_average(episode_durations, periods=5)
    mean_reward = moving_average(reward_in_episode, periods=5)
    lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
    lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon')
    lines.append(ax2.plot(epsilon_vec, label="epsilon", color="C3")[0])
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc=3)

    return fig


def train(episodes: int = 25000,
          lr: float = 0.01,
          gamma: float = 0.99,
          epsilon: float = 1,
          max_epsilon: float = 1,
          min_epsilon: float = 0.001,
          epsilon_decay: float = 0.01) -> tuple[float, int]:
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    col1, col2, col3 = st.columns(3)

    total_reward = []
    steps_per_episode = []
    epsilon_vec = []

    with col1:
        st.write("**{}** - Starting Training...\n".format(
            start_date.strftime("%Y/%m/%d - %H:%M:%S")))
    for e in range(episodes):
        state = env.reset()

        done = False
        total_reward.append(0)
        steps_per_episode.append(0)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * e)
        epsilon_vec.append(epsilon)

        # Loop as long as the game is not over, i.e. done is not True
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore the action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            # Apply the action and see what happens
            next_state, reward, done, _ = env.step(action)
            total_reward[e] += reward
            steps_per_episode[e] += 1

            current_value = q_table[
                state, action]  # current Q-value for the state/action couple
            next_max = np.max(q_table[next_state])  # next best Q-value

            # Compute the new Q-value with the Bellman equation
            q_table[state, action] = (1 - lr) * current_value + lr * (
                reward + gamma * next_max)
            state = next_state

    fig = plot_durations(steps_per_episode,
                         total_reward,
                         epsilon_vec,
                         max_steps_per_episode=200)

    end_date = datetime.now()
    execution_time = (time.time() - start_time)

    st.write()
    with col1:
        st.write("**{}** - Training Ended".format(
            end_date.strftime("%Y/%m/%d - %H:%M:%S")))
    with col2:
        st.write("Mean Reward:")
        st.write("**{}**".format(np.round(np.mean(total_reward), 2)))
    with col3:
        st.write("Time to train:")
        st.write(
            " **{}**s &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **{}**min &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **{}**h"
            .format(np.round(execution_time, 2),
                    np.round(execution_time / 60, 2),
                    np.round(execution_time / 3600, 2)))

    np.save("q-table", q_table)
    plt.savefig("Q-Learning_graph.png")

    st.write("Q-Table saved in *q-table.npy*")
    st.pyplot(fig)

    return np.round(execution_time, 2), np.mean(total_reward)


if __name__ == "__main__":
    st.title("Q Learning")
    st.header("Training a Model")
    env = gym.make("Taxi-v3")

    mode = st.selectbox("Select a Training Mode", ("Custom", "Performance"))

    if mode == "Custom":

        a, b, c = st.columns(3)

        with a:
            episodes = st.number_input("Episodes",
                                       min_value=1,
                                       step=1,
                                       value=25000,
                                       key="episode_1")
            epsilon = st.number_input("Exploration Rate",
                                      min_value=0.0,
                                      max_value=1.0,
                                      value=1.0,
                                      format="%.3f",
                                      key="epsilon_1")

        with b:
            lr = st.number_input("Learning Rate",
                                 min_value=0.0,
                                 max_value=1.0,
                                 value=0.01,
                                 format="%.3f",
                                 key="lr_1")
            min_eps = st.number_input("Min Exploration Rate",
                                      min_value=0.0,
                                      max_value=epsilon,
                                      step=0.001,
                                      value=0.001,
                                      format="%.3f",
                                      key="mineps_1")

        with c:
            gamma = st.number_input("Discount Rating",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.99,
                                    format="%.3f",
                                    key="gamma_1")
            decay = st.number_input("Decay Rate",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.01,
                                    format="%.3f",
                                    key="decay_1")
    else:
        episodes = 25000
        epsilon = 1.0
        lr = 0.01
        min_eps = 0.001
        gamma = 0.99
        decay = 0.01
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.write("Episodes")
            st.write(episodes)
        with col2:
            st.write("Exploration Rate")
            st.write(epsilon)
        with col3:
            st.write("Min Exploration Rate")
            st.write(min_eps)
        with col4:
            st.write("Learning Rate")
            st.write(lr)
        with col5:
            st.write("Discount Rating")
            st.write(gamma)
        with col6:
            st.write("Decay Rate")
            st.write(decay)

    training_started = False
    train_start = st.button("Start Training")

    if train_start:
        if not training_started:
            start_date = datetime.now()
            start_time = time.time()
            training_started = True

            env = gym.make("Taxi-v3")

            time, reward = train(episodes, lr, gamma, epsilon, epsilon,
                                 min_eps, decay)
            training_started = False
