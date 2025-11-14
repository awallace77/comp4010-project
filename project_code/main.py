from envs.tower_defense_env import TowerDefenseEnv
from rl.qlearn import q_learning, evaluate_policy
"""
    Main entry point for Tower defense RL
"""
if __name__ == "__main__":

    # env = TowerDefenseEnv(render_mode="human", render_rate=100) 
    env = TowerDefenseEnv(render_mode=None, num_enemies=3)
    
    # Approximate Q values
    Q = q_learning(env, 
                   episodes=10000, 
                   alpha=0.5, 
                   gamma=0.9, 
                   epsilon_start=0.9, 
                   epsilon_end=0.05,
                   epsilon_decay_steps=1500,
                   log=True)
    env.close()

    env = TowerDefenseEnv(render_mode="human", render_rate=200, num_enemies=3) 
    # evaluate_policy(env, Q, episodes=5, sleep=0.5)
    evaluate_policy(env, Q)
    env.close()