# Project Progress Update (NOV 30)

Group 12
- Andrew Wallace - 101210291 - andrewwallace3@cmail.carleton.ca
- Mohammad Rehman - 101220514 - mohammadrehman@cmail.carleton.ca
- Manal Hassan - 101263813 - manalhassa@cmail.carleton.ca
- Derrick Zhang - 101232374 - derrickzhang@cmail.carleton.ca

In the algorithm implementation phase (Nov 15 - Nov 30), we implemented multiple RL algorithms and created an evaluation framework to compare their performance.

## Algorithm Implementation

**Q-Learning with Neural Network** (Andrew Wallace, Manal Hassan)
- Implemented Q-Learning using a neural network for function approximation
- Added target network and Huber loss for training stability

**SARSA with Neural Network** (Andrew Wallace)
- Implemented SARSA with Q-network function approximation
- On-policy TD learning with epsilon-greedy exploration

**Deep Q-Network (DQN)** (Derrick Zhang, Manal Hassan)
- Implemented DQN with experience replay buffer
- Added target network updates for stable learning

**Advantage Actor-Critic (A2C)** (Andrew Wallace)
- Implemented A2C using tile coding for feature representation
- Softmax policy with linear function approximation

## Bug Fixes and Testing

**Action Mapping Fix** (Mohammad Rehman)
- Fixed bug in AoE tower placement where action-to-grid mapping was incorrect
- Actions 101-200 now correctly map to grid positions for AoE towers

**Test Suite** (Derrick Zhang)
- Added tests for single-target and AoE towers against enemy waves
- Created test to visualize enemy spawning and pathing to base
- Updated AoE tower to cover square area around tower

## Evaluation Framework

**Evaluation Module** (Mohammad Rehman)
- Created evaluation module to collect per-episode metrics (reward, waves completed, enemies killed, towers placed)
- Added algorithm comparison tools with aggregate statistics
- Implemented visualization for comparing algorithm performance

## Team Contributions
Each member contributed to testing, debugging, and validating the implementations.

## Plan for Final Week
- Run full evaluation across all algorithms
- Complete final report with algorithm analysis
- Record demo video showing trained agents
