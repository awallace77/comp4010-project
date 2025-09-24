# Proposal Brainstorming

## Traffic signal control

Source: https://ieeexplore.ieee.org/abstract/document/8667868

- Adaptive traffic signal control (ATSC)
- centralized RL is infeasible for large-scale ATSC b/c of high dimension of the joint action space
- multi-agent RL (MARL) overcomes this by distributing global control to each local RL agent; but now env becomes partially observable from the viewpoint of each local agent due to limited comms among agents
- this paper presents advantage actor critic (A2C) algorithm within the context of ATSC
- Proposed multi-agent A2C is compared against independent A2C and independent Q-learning algos & results show optimality, robustness and sample efficiency

### Summary

Quite a high complexity and math heavy with A2C algo

## Learning a Game

Source: Reinforcement Learning in Games - https://link.springer.com/chapter/10.1007/978-3-642-27645-3_17

- Looks at Backgammon, Chess, Go, Dyna-2, Tetris, RTS games (war-simulations)

### Ideas

- Single or Multi-agent snake game with obstacles / enemies to avoid (like pac-man)

### Search and Rescue Helicopter Game

- Train an RL agent to control a helicopter navigate a 2D env to find and rescue survivors while avoiding obstacles, limited fuel, and time constraints
- Goal find and rescue the survivors

#### State Space

- Helicopter position, velocity, and angle
- Remaining fuel
- Known map areas vs. unexplored areas (fog of war)
- Positions of Hazards (e.g., wind gusts, falling rocks, enemies?)
- Survivor's location (revealed only when discovered)

#### Action space

- Continuous control: thrust, rotate, move up/down/left/right
- OR discrete control (up/down/left/right/hover/rescue) which is simpler

#### Reward structure

- +100 for rescuing a survivor
- -10 for collision or crashing or being shot by enemy or smth
- -1 per timestep (to ensure finds survivor in a timely manner)
- -5 for running out of fuel

#### Novelty

- Dynamic hazards such as wind gusts, bazooka missile, moving birds / drones, etc.
- Multi agent extension - add an additional helicopter or have survivors act as agents to get to a better position?
- Hierarchial RL - high level policy for deciding search vs rescue; low level policy for handling navigation
- Partially observable env - survivor location unknown until discovered (immediate next state) thus agent must explore

##### Easiest high-impact improvements:

1. Add fog of war to make it a partially observable problem.
2. Use multi-agent cooperation or hierarchical RL.
3. Introduce dynamic hazards for non-stationarity.

#### Environment Implementation

- PyGame → good for a simple 2D rescue scenario.

- OpenAI Gymnasium Custom Env → easy integration with existing RL libraries like Stable Baselines 3.

- Physics can be kept simple like Lunar Lander or more complex with libraries like PyBullet.
