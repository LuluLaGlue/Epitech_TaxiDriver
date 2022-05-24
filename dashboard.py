import gym
import time
import torch
import pickle
import random
import datetime
import numpy as np
import streamlit as st
import torch.optim as optim
import matplotlib.pyplot as plt
from DQN.DQN import DQN, DQN_2
from DQN.DQN_Play import play as play_DQN
from SARSA.SARSA_Play import solve as play_S
from MonteCarlo.MC_Play import solve as play_MC
from Bruteforce.Bruteforce import play as play_BF
from ValueIteration.VI_Play import solve as play_VI
from QLearning.QLearning_Play import solve as play_QL

st.set_page_config(layout="wide")


def import_model(env, path: str) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path)

    n_actions = env.action_space.n
    n_observation = env.observation_space.n
    model = DQN(n_observation, n_actions).to(
        device) if checkpoint.get("architecture") == 1 or checkpoint.get(
            "architecture") == None else DQN_2(n_observation,
                                               n_actions).to(device)
    optimizer = optim.RMSprop(model.parameters())

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer, model, device


def display_data(total, total_failed, start, mean_steps, mean_result):
    return "[**{}** LOOP DONE - **{}%** FAILED - **{}** SECONDES] - Mean Steps Per Loop: **{}** - Mean Reward Per Loop: **{}**".format(
        total, np.round(total_failed / total * 100, 2),
        np.round(time.time() - start, 4), np.round(mean_steps / total, 2),
        np.round(mean_result / total, 2))


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

    end_date = datetime.datetime.now()
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

    np.save("QLearning/qtable", q_table)
    plt.savefig("QLearning/QLearning_graph.png")

    st.write("qtable saved in *qtable.npy*")
    st.pyplot(fig)

    return np.round(execution_time, 2), np.mean(total_reward)


if __name__ == "__main__":
    st.title("Q Learning")
    st.header("Test an algorithm")
    env = gym.make("Taxi-v3")

    algo = st.selectbox("Select an algorithm",
                        ("Bruteforce", "Value Iteration", "Monte Carlo",
                         "Q-Learning", "SARSA", "DQN"))
    a, b = st.columns(2)
    with a:
        time_to_run = st.number_input(
            "Time to run (seconds) - set to 0 in order to use loops",
            min_value=0,
            step=1)
    with b:
        nbr_loop = st.number_input("Number of loop to do", min_value=1, step=1)
    start_playing = st.button("Play")
    mean_steps, mean_result, total_failed = 0, 0, 0
    max_steps = []
    is_loop = True if nbr_loop != 1 else False
    playing = False

    if not playing:
        playing = True
        if algo == "Bruteforce" and start_playing:
            start = time.time()
            maxrt = datetime.timedelta(
                seconds=time_to_run) if time_to_run != 0 else None

            env = gym.make("Taxi-v3")
            if maxrt != None:
                st.write("**{}** - Playing for {} seconds...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    time_to_run))
                stop = datetime.datetime.now() + maxrt
                total = 0
                while datetime.datetime.now() < stop:
                    steps, result = play_BF(env=env,
                                            is_loop=is_loop,
                                            is_time=True)
                    max_steps.append(steps)
                    mean_steps += steps
                    mean_result += result
                    total += 1
                st.write(display_data(total, 0, start, mean_steps,
                                      mean_result))
            else:
                st.write("**{}** - Playing {} loops...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    nbr_loop))
                for l in range(nbr_loop):
                    steps, result = play_BF(env=env,
                                            is_loop=is_loop,
                                            is_time=False)
                    max_steps.append(steps)
                    mean_steps += steps
                    mean_result += result
                st.write(
                    display_data(nbr_loop, 0, start, mean_steps, mean_result))
        elif algo == "Value Iteration" and start_playing:
            start = time.time()

            maxrt = datetime.timedelta(
                seconds=time_to_run) if time_to_run != 0 else None

            if maxrt != None:
                st.write("**{}** - Playing for {} seconds...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    time_to_run))
                stop = datetime.datetime.now() + maxrt
                total = 0

                while datetime.datetime.now() < stop:
                    mean_steps, mean_result, total_failed = play_VI(
                        "ValueIteration/v-iteration.npy", False, False,
                        mean_steps, mean_result, total_failed, is_loop, True)
                    total += 1

                st.write(
                    display_data(total, total_failed, start, mean_steps,
                                 mean_result))
            else:
                st.write("**{}** - Playing {} loops...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    nbr_loop))
                for l in range(nbr_loop):
                    mean_steps, mean_result, total_failed = play_VI(
                        "ValueIteration/v-iteration.npy", False, False,
                        mean_steps, mean_result, total_failed, is_loop, False)

                st.write(
                    display_data(nbr_loop, total_failed, start, mean_steps,
                                 mean_result))
        elif algo == "Monte Carlo" and start_playing:
            with open('MonteCarlo/policy.pkl', 'rb') as f:
                policy = pickle.load(f)
            start = time.time()

            maxrt = datetime.timedelta(
                seconds=time_to_run) if time_to_run != 0 else None

            if maxrt != None:
                st.write("**{}** - Playing for {} seconds...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    time_to_run))
                stop = datetime.datetime.now() + maxrt
                total = 0

                while datetime.datetime.now() < stop:
                    mean_steps, mean_result, total_failed = play_MC(
                        policy,
                        mean_steps,
                        mean_result,
                        total_failed,
                        slow=False,
                        render=False,
                        is_time=True,
                        is_loop=is_loop)
                    total += 1
                st.write(
                    display_data(total, total_failed, start, mean_steps,
                                 mean_result))
            else:
                st.write("**{}** - Playing {} loops...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    nbr_loop))
                for l in range(nbr_loop):
                    mean_steps, mean_result, total_failed = play_MC(
                        policy,
                        mean_steps,
                        mean_result,
                        total_failed,
                        slow=False,
                        render=False,
                        is_time=False,
                        is_loop=is_loop)
                st.write(
                    display_data(nbr_loop, total_failed, start, mean_steps,
                                 mean_result))
        elif algo == "Q-Learning" and start_playing:
            start = time.time()

            maxrt = datetime.timedelta(
                seconds=time_to_run) if time_to_run != 0 else None

            if maxrt != None:
                st.write("**{}** - Playing for {} seconds...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    time_to_run))
                stop = datetime.datetime.now() + maxrt
                total = 0
                while datetime.datetime.now() < stop:
                    mean_steps, mean_result, total_failed = play_QL(
                        "QLearning/qtable.npy", mean_steps, mean_result,
                        total_failed, False, False, is_loop, True)
                    total += 1

                st.write(
                    display_data(total, total_failed, start, mean_steps,
                                 mean_result))
            else:
                st.write("**{}** - Playing {} loops...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    nbr_loop))
                for l in range(nbr_loop):
                    mean_steps, mean_result, total_failed = play_QL(
                        "QLearning/qtable.npy", mean_steps, mean_result,
                        total_failed, False, False, is_loop, False)

                st.write(
                    display_data(nbr_loop, total_failed, start, mean_steps,
                                 mean_result))
        elif algo == "SARSA" and start_playing:
            start = time.time()

            maxrt = datetime.timedelta(
                seconds=time_to_run) if time_to_run != 0 else None

            if maxrt != None:
                st.write("**{}** - Playing for {} seconds...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    time_to_run))
                stop = datetime.datetime.now() + maxrt
                total = 0

                while datetime.datetime.now() < stop:
                    mean_steps, mean_result, total_failed = play_S(
                        "SARSA/qtable.npy", False, False, mean_steps,
                        mean_result, total_failed, is_loop, True)
                    total += 1

                st.write(
                    display_data(total, total_failed, start, mean_steps,
                                 mean_result))
            else:
                st.write("**{}** - Playing {} loops...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    nbr_loop))
                for l in range(nbr_loop):
                    mean_steps, mean_result, total_failed = play_S(
                        "SARSA/qtable.npy", False, False, mean_steps,
                        mean_result, total_failed, is_loop, False)

                st.write(
                    display_data(nbr_loop, total_failed, start, mean_steps,
                                 mean_result))
        elif algo == "DQN" and start_playing:
            start = time.time()
            _, model, device = import_model(
                env, "DQN/models/reference/DQN_reference.pt")

            maxrt = datetime.timedelta(
                seconds=time_to_run) if time_to_run != 0 else None

            if maxrt != None:
                st.write("**{}** - Playing for {} seconds...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    time_to_run))
                stop = datetime.datetime.now() + maxrt
                total = 0
                while datetime.datetime.now() < stop:
                    steps, result, mean_steps, mean_result = play_DQN(
                        env,
                        model,
                        100,
                        mean_steps,
                        mean_result,
                        total_failed,
                        render=False,
                        slow=False,
                        is_time=True,
                        is_loop=is_loop,
                        device=device)
                    total += 1
                st.write(
                    display_data(total, total_failed, start, mean_steps,
                                 mean_result))
            else:
                st.write("**{}** - Playing {} loops...\n".format(
                    datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                    nbr_loop))
                for l in range(nbr_loop):
                    steps, result, mean_steps, mean_result = play_DQN(
                        env,
                        model,
                        100,
                        mean_steps,
                        mean_result,
                        total_failed,
                        render=False,
                        slow=False,
                        is_time=False,
                        is_loop=is_loop,
                        device=device)

                st.write(
                    display_data(nbr_loop, total_failed, start, mean_steps,
                                 mean_result))
        playing = False
    st.header("Training a Q-Learning Model")

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
            start_date = datetime.datetime.now()
            start_time = time.time()
            training_started = True

            env = gym.make("Taxi-v3")

            time, reward = train(episodes, lr, gamma, epsilon, epsilon,
                                 min_eps, decay)
            training_started = False
