import gym
import numpy as np
import tensorflow as tf

env = gym.make("FrozenLake-v1")

n_actions = env.action_space.n
input_dim = env.observation_space.n
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3, input_dim=input_dim, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(n_actions, activation='linear'))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')


def replay(replay_memory, minibatch_size=32):
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
    s_l = np.array(list(map(lambda x: x['s'], minibatch)))
    a_l = np.array(list(map(lambda x: x['a'], minibatch)))
    r_l = np.array(list(map(lambda x: x['r'], minibatch)))
    sprime_l = np.array(list(map(lambda x: x['sprime'], minibatch)))
    done_l = np.array(list(map(lambda x: x['done'], minibatch)))
    qvals_sprime_l = model.predict(sprime_l)
    target_f = model.predict(s_l)
    for i, (s, a, r, qvals_sprime,
            done) in enumerate(zip(s_l, a_l, r_l, qvals_sprime_l, done_l)):
        if not done: target = r + gamma * np.max(qvals_sprime)
        else: target = r
        target_f[i][a] = target
    model.fit(s_l, target_f, epochs=1, verbose=0)
    return model


def encode(data, states_total):
    targets = np.array(data).reshape(-1)
    return np.eye(states_total)[targets]


n_episodes = 1000
gamma = 0.99
epsilon = 0.9
minibatch_size = 32
r_sums = []
replay_memory = []
mem_max_size = 100000

for n in range(n_episodes):
    ss = env.reset()
    states_total = 16
    data = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

    m = encode(data, states_total)
    s = m[ss]
    #print(s)
    #print(len(s))
    done = False
    r_sum = 0
    while not done:
        #env.render()
        qvals_s = model.predict(s.reshape(1, -1))
        if np.random.random() < epsilon: a = env.action_space.sample()
        else: a = np.argmax(qvals_s)
        sprime, r, done, info = env.step(a)
        r_sum += r
        q = encode(data, states_total)
        sprime = q[sprime]
        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)
        replay_memory.append({
            "s": s,
            "a": a,
            "r": r,
            "sprime": sprime,
            "done": done
        })
        #s = n[sprime]
        s = sprime
        model = replay(replay_memory, minibatch_size=minibatch_size)
    # env.render()
    if epsilon > 0.001: epsilon -= 0.001
    r_sums.append(r_sum)
    if n % 10 == 0:
        print("EPOCH {}: ".format(n), np.mean(r_sums), r_sums[-10:])

# print("Rewards: ", r_sums)
print("Mean: ", np.mean(r_sums))
