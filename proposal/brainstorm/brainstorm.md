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
