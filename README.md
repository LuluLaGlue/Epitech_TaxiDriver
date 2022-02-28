# Taxi Driver

This repository contains our attempt at Reinforcement Learning, the goal of this project is to train an AI capable of picking up a passanger in a random spot and driving him to another random spot on a map.

## Value Iteration Algorithm
One of the first algorithm used for reinforcement learning was the *Value Iteration Algorithm*, its core idea is to calculate the value of each state. It loops over all states and possible actions to explore rewards of a given action and calculates the maximum possible action/reward and stores it in a table. This solution can be found in the *Value Iteration* directory with the following files:
- ***VI_Train.py***: A python script used to train and save a new table (or model). To run it, use <code>python VI_Train.py</code> with the following possible arguments:
    + <code>--gamma</code>: Gamma or Discount Rating. **Default**: 0.9
    + <code>-h</code>: Display a help message.
- ***v-iteration.npy***: The resulting table (or model) stored in a *numpy* file. It took 1.22 secondes to train the one present.
- ***VI_Play.py***: A python script used to play the Taxi Game based on the priously trained model (***v-iteration.npy***). To run it use <code>python VI_Play.py</code> with the following possible arguments:
    + <code>-s</code>: Activate Slow Mode.
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game.
    + <code>-h</code>: Display a help message.
## Q Learning
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
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game (equivalent to *episodes* during training)
    + <code>-h</code>: Display a help message.

## SARSA
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
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game (equivalent to *episodes* during training)
    + <code>-h</code>: Display a help message.

## DQN
In order to train a model capable of accomplishing more complexe tasks we turned to Deep Q Learning. This attempt is located in the *DQN* directory with the following files:
- ***DQN_Train.py***: A python script used to train and save a new model. To run it use <code>python DQN_Train.py</code> with the following possible arguments:
    + <code>--environment</code>.Environment in which to train the model. **Default**: Taxi-v3
    + <code>--batch_size</code>. Batch Size during training. **Default**: 128
    + <code>--gamma</code>. Gamma. **Default**: 0.99
    + <code>--eps_start</code>. Starting Epsilon Value. **Default**: 1
    + <code>--eps_end</code>. Minimal Epsilon Value. **Default**: 0.1
    + <code>--eps_decay</code>. Decay Rate for Epsilon Value. **Default**: 400
    + <code>--target_update</code>. Number of episodes between each model update. **Default**: 20
    + <code>--max_steps</code>: Max Steps Allowed per Episode. **Default**: 100
    + <code>--warmup_episode</code>: Number of warmup episodes. **Default**: 20
    + <code>--save_freq</code>: Number of episodes between each model saving. **Default**: 1000
    + <code>--lr</code>: Initial Learning Rate. **Default**: 0.001
    + <code>--lr_min</code>: Minimal Learning Rate: **Default**: 0.0001
    + <code>--lr_decay</code>: Learning Rate Decay Speed: **Default**: 5000
    + <code>--memory</code>: Size of Memory. **Default**: 50000
    + <code>--name</code>: Name to give the resulting model. **Default**: Timestamp
    + <code>--model</code>: Path to a previously created model to further train (When resuming training, will take the previously set parameters)
    + <code>--episodes</code>: Number of episodes during training. **Default**: 10000
    + <code>--architecture</code>: Architecture to use (1 or 2): **Default**: 1
    + <code>-h</code>: Display a help message.
- ***DQN.py***: A python Class that contains our DQN models (DQN and DQN_2).
- ***models/***: This folder contains a few models trained using the ***DQN_Train.py*** with varying parameters.
- ***DQN_Play.py***: A python script used to play the Taxi Game based on a previously trained model (***models/***). To run it, use <code>python DQN_Play.py</code> with the following possible arguments:
    + <code>-p</code>: Path to model to use. **Default**: <code>./models/reference/DQN_reference.pt</code>
    + <code>-s</code>: Activate Slow Mode.
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game (equivalent to *episodes* during training). **Default**: 1
    + <code>-h</code>: Display a help message.
