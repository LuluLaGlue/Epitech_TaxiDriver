# **Reinforcement Learning Informations**

In order to train a proper model we need to define a few key concepts

## ***Environnement***
The agent is placed inside a map composed of icy tiles and holes, he needs to find the best/only path towards the end of the map without falling inside a hole. For each actions it takes there is a probability to slip in a different direction
### _States_
The position of the agent; An integer from _0_ to _number of tiles_ representing the position of the agent on the map.
### _Actions_
The agent can move in _4_ directions: _up_, _down_, _right_, _left_.
### _Rewards_
A positive reward is only given once the agent hs completed the objective (finishing the map without falling into a hole)
