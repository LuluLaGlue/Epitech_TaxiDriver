# Taxi Driver

This repository contains our attempt at Reinforcement Learning, the goal of this project is to train an AI capable of picking up a passanger in a random spot and driving him to another random spot on a map.

## First Approach
### Q Learning
In order to train a model capable of completing the task at hand we decided to use the Q Learning algorithm. This attempt is located in the *Q-Learning* directory with the following files:
- ***Q-Learning_Train.py***: A python script used to train and save a new model (or Q Table) to run it use <code>python Q-Learning_Train.py</code> with the following possible arguments:
    + <code>-l</code>: Learning Rate. **Default**: 0.99
    + <code>-g</code>: Gamma or Discount Rating. **Default**: 0.99
    + <code>--episodes</code>: Number of episodes to run during training. **Default**: 25000
    + <code>-e</code>: Epsilon or Exploration Rate. **Default**: 1
    + <code>--min_epsilon</code>: Minimal value for exploration rate. **Default**: 0.001
    + <code>-d</code>: Decay Rate. **Default**: 0.01
    + <code>-h</code>: Display a help message.
- ***q-table.npy***: The resulting model (or Q Table) stored in a *numpy* file. It took 17 seconds to train it during *25 000* episodes.
- ***Q-Learning_Play.py***: A python script used to play the Taxi Game based on the previously trained model (***q-table.npy**) to run it use <code>python Q-Learning_Play.py</code> with the following possible arguments:
    + <code>-s</code>: Activate Slow Mode.
    + <code>-h</code>: Display a help message.

### SARSA
In order to shorten the training time we tried to implement the SARSA algorithm. This attempt is located in the *SARSA* directory with the following files:
- ***SARSA_Train.py***: A python script used to train and save a new model (or Q Table), to run it use <code>python SARSA_Train.py</code> with the following possible arguments:
    + <code>-a</code>: Alpha. **Default**: 0.85
    + <code>-g</code>: Gamma or Discount Rating. **Default**: 0.99 (We first tried with *0.95* but the resulting model was not effective enough)
    + <code>--episodes</code>: Number of episodes to run during training. **Default**: 10000
    + <code>-e</code>: Epsilon or Exploration Rate. **Default**: 1
    + <code>--min_epsilon</code>: Minimal value for exploration rate. **Default**: 0.001
    + <code>-d</code>: Decay Rate. **Default**: 0.01
    + <code>-h</code>: Display a help message.
- ***q-table.npy***: The resulting model (or Q Table) stored in a *numpy* file. We observed that the *25 000* episodes used with ***Q Learning*** were no longer need and that the model could be trained with roughly *2 000* episodes, doing so took only 1.22s of training.
- ***SARSA_Play.py***: A python script used to play the Taxi Game based on the previously trained model (***q-table.npy***) to run it use <code>python SARSA_Play.py</code> with the following possible arguments:
    + <code>-s</code>: Activate Slow Mode.
    + <code>-h</code>: Display a help message.