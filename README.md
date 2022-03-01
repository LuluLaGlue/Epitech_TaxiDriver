# Taxi Driver
&emsp;This repository contains our attempt at Reinforcement Learning, the goal of this project is to train an AI capable of picking up a passanger in a random spot and driving him to another random spot on a map. To do so we worked with *OpenAI*'s *gym*'s *Taxi* environment.

&emsp;In this environment there are 4 different locations indicated by *R(ed)*, *B(lue)*, *Y(ellow)*, *G(reen)*. At the start of an episode, the taxi starts off at a random square and the passenger is at a random location (*R*, *B*, *Y* or *G*). The taxi must drive to the passenger's location, pick him up, drive him to his destination (another random location) and drop him off. Once the passenger is dropped off the episode ends. The Taxi can not go through walls.

&emsp;There are *6* possible actions:
- *0*: Go *South* (or *down*),
- *1*: Go *North* (or *up*),
- *2*: Go *East* (or *right*),
- *3*: Go *West* (or *left*),
- *4*: *Pickup* the passenger,
- *5*: *Dropoff* the passenger.

&emsp;Each action taken by the agent will result in a reward:
- *Moving* results in a **-1** reward, this will guide the agent towards a more efficient path between objectives,
- *picking up* or *dropping off* at a forbidden location results in a **-20** reward, this will guide the agent towards not *spamming* the *dropoff* and *pickup* actions,
- *dropping off* the passenger at the correct location results in a **+20** reward, this will guide the agent towards completing the objective and is the only positive reward available.

# Algorithms

## Value Iteration Algorithm
&emsp;One of the first algorithm used for reinforcement learning was the *Value Iteration Algorithm*, its core idea is to calculate the value of each state. It loops over all states and possible actions to explore rewards of a given action and calculates the maximum possible action/reward and stores it in a table. This solution can be found in the *Value Iteration* directory with the following files:
- ***VI_Train.py***: A python script used to train and save a new table (or model). To run it, use <code>python VI_Train.py</code> with the following possible arguments:
    + <code>--gamma</code>: Gamma or Discount Rating. **Default**: 0.9
    + <code>-h</code>: Display a help message.
- ***v-iteration.npy***: The resulting table (or model) stored in a *numpy* file. It took 1.22 secondes to train the one present.
- ***VI_Play.py***: A python script used to play the Taxi Game based on the priously trained model (***v-iteration.npy***). To run it use <code>python VI_Play.py</code> with the following possible arguments:
    + <code>-s</code>: Activate Slow Mode.
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game.
    + <code>-h</code>: Display a help message.

&emsp;Per its trainig logic this approach is is model based as we have to know all environment states/transitions upfront so the algorithm works.

## Q Learning
&emsp;In order to train a model capable of completing the task at hand without being model based we decided to use the Q Learning algorithm. This algorithm is centred around the actor (the Taxi in our case) and starts exploring on trial-and-error to update its knowledge about the model (= path to the best reward). During training, this algorithm will update a matrice containing the maximum discounted future reward for each action and state. It is based on the Bellman equation extended with a learning rate (we set it by default to *0.01* with gamma at *0.99*):

<code>New Q(S, A) = Q(S, A) + alpha * (R(S, A) + gamma * MaxQ'(S', A') - Q(S, A))</code>.

&emsp;This attempt is located in the *Q-Learning* directory with the following files:
- ***Q-Learning_Train.py***: A python script used to train and save a new model (or Q-Matrix) to run it use <code>python Q-Learning_Train.py</code> with the following possible arguments:
    + <code>-l</code>: Learning Rate. **Default**: 0.01
    + <code>-g</code>: Gamma or Discount Rating. **Default**: 0.99
    + <code>--episodes</code>: Number of episodes to run during training. **Default**: 25000
    + <code>-e</code>: Epsilon or Exploration Rate. **Default**: 1
    + <code>--min_epsilon</code>: Minimal value for exploration rate. **Default**: 0.001
    + <code>-d</code>: Decay Rate. **Default**: 0.01
    + <code>-h</code>: Display a help message.
- ***q-table.npy***: The resulting model (or Q-Matrix) stored in a *numpy* file. It took 17 seconds to train it during *25 000* episodes.
- ***Q-Learning_Play.py***: A python script used to play the Taxi Game based on the previously trained model (***q-table.npy**) to run it use <code>python Q-Learning_Play.py</code> with the following possible arguments:
    + <code>-s</code>: Activate Slow Mode.
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game (equivalent to *episodes* during training)
    + <code>-h</code>: Display a help message.

&emsp;This approach is effective in our case as the only positive reward is the correct dropoff of the passenger. If the environment were to contain another positive reward the trial-and-error approach might optimize the route to it and miss out on the real goal of the game it is learning to play. In order to to limit this we implemented the Epsilon Decreasing method which consists of exploiting the current situation with probability <code>1 - epsilon</code> and exploring a new option with probability <code>epsilon</code> with epsilon decreasing over time. The Epsilon Decreasing method is particularly effective in environment such as Frozen Lakes where the game actions are not deterministic.

## SARSA
&emsp;In order to shorten the training time and explore other possible algorithm we tried to implement the SARSA algorithm. This attempt is located in the *SARSA* directory with the following files:
- ***SARSA_Train.py***: A python script used to train and save a new model (or Q-Matrix), to run it use <code>python SARSA_Train.py</code> with the following possible arguments:
    + <code>-a</code>: Alpha. **Default**: 0.85
    + <code>-g</code>: Gamma or Discount Rating. **Default**: 0.99 (We first tried with *0.95* but the resulting model was not effective enough)
    + <code>--episodes</code>: Number of episodes to run during training. **Default**: 10000
    + <code>-e</code>: Epsilon or Exploration Rate. **Default**: 1
    + <code>--min_epsilon</code>: Minimal value for exploration rate. **Default**: 0.001
    + <code>-d</code>: Decay Rate. **Default**: 0.01
    + <code>-h</code>: Display a help message.
- ***q-table.npy***: The resulting model (or Q-Matrix) stored in a *numpy* file. We observed that the *25 000* episodes used with ***Q Learning*** were no longer need and that the model could be trained with roughly *2 000* episodes, doing so took only 1.22s of training.
- ***SARSA_Play.py***: A python script used to play the Taxi Game based on the previously trained model (***q-table.npy***) to run it use <code>python SARSA_Play.py</code> with the following possible arguments:
    + <code>-s</code>: Activate Slow Mode.
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game (equivalent to *episodes* during training)
    + <code>-h</code>: Display a help message.

## DQN
&emsp;In order to train a model capable of accomplishing more complexe tasks we turned to Deep Q Learning. Indeed, the algorithm showcased before are perfect for small environment such as the Taxi driver or Frozen Lake but in more complexe environment with a lot more observation space (an Atari Video Game for exemple) they will quickly be unmanageable. Furthermore, the Q-Agent has no ability to estimate value for an unseen state, it will go back at best to to random actions. To deal with this problem *Deep Q Learning* algorithm remove the two dimensional Q-Matrix and replace it with a Neural Network. This attempt is located in the *DQN* directory with the following files:
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
    + <code>--architecture</code>: Architecture to use (1 or 2): **Default**: 2
    + <code>-h</code>: Display a help message.
- ***DQN.py***: A python Class that contains our DQN models (DQN and DQN_2).
- ***models/***: This folder contains a few models trained using the ***DQN_Train.py*** with varying parameters.
- ***DQN_Play.py***: A python script used to play the Taxi Game based on a previously trained model (***models/***). To run it, use <code>python DQN_Play.py</code> with the following possible arguments:
    + <code>-p</code>: Path to model to use. **Default**: <code>./models/reference/DQN_reference.pt</code>
    + <code>-s</code>: Activate Slow Mode.
    + <code>-r</code>: Activate Render.
    + <code>-l</code>: Set a number of times to play the game (equivalent to *episodes* during training). **Default**: 1
    + <code>-h</code>: Display a help message.
- ***DQN.ipynb***: A jupyter notebook demonstrating the process to train a new model as well as the different parameters we applied to train different models with different parameters and there impact on training graphs (*epsilon*, *steps* and *reward* per episode).
- ***models.csv***: Since trained several models to compare how each parameter influenced performances we saved each model's parameters along with its win rate in this *.csv* file.

# Conclusions

&emsp;For every algorithm/approach we tried, we manage to setup a fonctionnal agent capable of effectively completing the game.
- We started with the Value Iteration algorithm which took *48.96* secondes to solve the game *10 000* times with an average *14.06* steps and *7.94* reward per episode and a *100%* win rate.
- We then used the *Q-Learning* algorithm based on a Q-Matrix and trained it with several parameters (we settled with <code>gamma = 0.99</code>, <code>Learning Rate = 0.01</code>, <code>minimal epsilon = 0.001</code> and <code>Decay Rate = 0.01</code>), the resulting model takes *50.32* secondes to solve the game *10 000* times with an average *15.35* steps and *6.33* reward per episode and a *98.48%* win rate.
- In order to improve those metrics we used the *SARSA* algorithm and tuned it in the same way as the *Q-Learning* one (we settled with <code>alpha = 0.85</code>, <code>gamma = 0.99</code>, <code>minimal epsilon = 0.001</code> and <code>Decay Rate = 0.01</code>) we managed to solve the game *10 000* times in *49.43* secondes with an average *16.18* steps and *-7.13* reward per episode and a *98.56%* win rate.
- Finally, we trained 2 *Deep Q Learning* algorithm based on 2 different architectures, after fine tuning the parameters we settled with <code>batch_size = 128</code>, <code>gamma = 0.99</code>, <code>minimal epsilon = 0.1</code>, <code>epsilon decay = 400</code>, <code>number of episodes betzeen each model update = 20</code>, <code>max number of steps per episode = 100</code>, <code>save frequency = 1000</code>, <code>learning rate = 0.001</code>, <code>minimal learning rate = 0.0001</code>, <code>learning rate decay = 5000</code>, <code>memory size = 50000</code>, <code>architecture = 2</code>. With the best architecture and parameters we trained we achieved *10 000* games in *17.82* secondes with an average *13.10* steps and *7.90* reward per episode and a *100%* win rate.

&emsp;With the resulting metrics in mind we can clearly conclude that a classic *Machine Learning* approach (Q-Learning, SARSA, etc.) is working in a small environment such as the Frozen Lake but in a more complexe environment *Deep Learning* is required to attain a sufficient win rate in a limited time. The environment in which we trained our models (Taxi-v3) is at the limit between small enough to use *Machine Learning* and big enough to benefit from *Deep Learning*, we clearly outlined the limitations of *Machine Learning* with bellow *99%* winrate in both *Q-Learning* and *SARSA* but still managed to solve the game in acceptable times, *Deep Learning* shined with both perfect win rate (*100%*) and short execution times (*17.82* secondes for *10 000* loops).

&emsp;On a side note, even though *Value Iteration* obtained a *100%* winrate it took almost *50* secondes to clear the *10 000* loops which, compared to the *18* secondes of *DQN* is way to high.