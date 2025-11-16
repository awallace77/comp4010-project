# Project Progress Update

Group 12

- Andrew Wallace - 101210291 - andrewwallace3@cmail.carleton.ca
- Mohammad Rehman - 101220514 - mohammadrehman@cmail.carleton.ca
- Manal Hassan - 101263813 - manalhassa@cmail.carleton.ca
- Derrick Zhang - 101232374 - derrickzhang@cmail.carleton.ca

In the training and initial results phase (Oct 30 - Nov 15), we made  progress in enhancing the environment with new features and beginning research on chosen algorithms.

## Environment Enhancements

**Base Health and Tower Placement** (Mohammad Rehman)
- Implemented base health logic and UI visualization to track the player's remaining lives
- Added tower placement restrictions to ensure towers can only be placed in designated tower areas

**Code Refactoring** (Andrew Wallace)
- code restructuring for improved organization, reusability, and extensibility
- Created modular architecture with separate directories:
  - `envs/` for the gymnasium environment
  - `game/` for game logic and entities (base, enemy, tower)
  - `rl/` for reinforcement learning algorithms

**Tower Evolution and Area-of-Effect Tower Enhancements** (Andrew Wallace)
- Implemented a tower experience and leveling system: towers now gain experience from defeating enemies and level up after reaching set thresholds
- Tower statistics (such as damage and range) scale with tower level, with visual indicators reflecting upgraded towers
- Introduced an Area-of-Effect (AoE) tower type alongside single-target towers, featuring splash damage mechanics to attack multiple enemies
- Updated the state space to account for both tower levels and different tower types
- Balanced AoE tower attributes (damage, attack speed, cost) relative to single-target towers 

## Algorithm Research

**Deep Q-Learning Research** (Manal Hassan)
- Conducted initial research on Deep Q-Learning algorithms for tower defense
- Began exploring neural network architectures suitable for the expanded state space

## Team Contributions

Each member contributed to testing, debugging, and validating the new features. 

## Plan for Next Two Weeks

### Algorithm Implementation and Training (Nov 11–Nov 24)

- Implement Deep Q-Learning with neural network function approximation
- Begin training sessions with the enhanced environment featuring tower evolution and AoE towers
- Collect and analyze performance metrics comparing tabular Q-Learning vs. Deep Q-Learning
and other variants of the Deep Q-learning algorithms.
- Investigate A2C (Advantage Actor-Critic)

