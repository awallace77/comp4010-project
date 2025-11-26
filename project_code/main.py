"""
    Main entry point for Tower Defense RL
"""
from envs.tower_defense_env import TowerDefenseEnv
from rl.qlearn import q_learning, evaluate_policy

# CONFIGURATION
# =============

ALGORITHM = "Q_LEARNING"

# ALGORITHM PARAMETERS
# ====================

Q_LEARNING_PARAMS = {
    "episodes": 100,
    "alpha": 0.5,
    "gamma": 0.9,
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 1500,
    "log": True
}

# TODO: Add DQN_PARAMS when implemented
# TODO: Add PPO_PARAMS when implemented

if __name__ == "__main__":
    
    # Q-LEARNING
    if ALGORITHM == "Q_LEARNING":
        # Training
        env = TowerDefenseEnv(render_mode=None, num_enemies=3, size=10)
        Q = q_learning(env, **Q_LEARNING_PARAMS)
        env.close()

        # Evaluation
        env = TowerDefenseEnv(render_mode="human", render_rate=200, num_enemies=5, size=10)
        evaluate_policy(env, Q)
        env.close()

    # DQN

    # PPO

    else:
        print(f"ERROR: Unknown algorithm '{ALGORITHM}'")
        print("Available: Q_LEARNING")
