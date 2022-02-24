# **Reinforcement Learning Informations**

In order to train a proper model we need to define a few key concepts

## ***Environnement***
Our model will interact with a specific environnement composed of:
+ The *taxi* it will control,
+ A *passenger* it has to pickup,
+ A *destination* to which it must drop the passenger.
In order to reach the passenger then the destination it will need to move between tiles.
### _States_
Each action taken by the model will resulting in a change in the environment *states*. The states are a representation of the *taxi*'s position in the map as well as the *passenger* and the *destination*.
### _Actions_
At each step, the agent can take one of **6** actions:
- *0*: Go *South* (or *down*),
- *1*: Go *North* (or *up*),
- *2*: Go *East* (or *right*),
- *3*: Go *West* (or *left*),
- *4*: *Pickup* the passenger,
- *5*: *Dropoff* the passenger.
### _Rewards_
Each action taken by the agent will result in a reward:
- *Moving* results in a **-1** reward, this will guide the agent towards more efficient path between objectives,
- *picking up* or *dropping off* at a forbidden location results in a **-20** reward, this will guide the agent towards not *spamming* the *dropoff* and *pickup* actions,
- *dropping off* the passenger at the correct location results in a **+20** reward, this will guide the agent towards completing the objective.
