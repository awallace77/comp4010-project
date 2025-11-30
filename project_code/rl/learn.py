from envs.tower_defense_env import TowerDefenseEnv
import time
import numpy as np
import gymnasium as gym

from rl.utils import evaluate_policy_fa, evaluate_policy_nn, render_env
from rl.featurizers.tile_coder_featurizer import TileCoder

# Algo imports
from rl.qlearn import q_learning, evaluate_policy
from rl.a2c import a2c, softmaxPolicy, run_a2c_experiments 
from rl.sarsa import sarsa, run_sarsa_experiments, greedy_policy as sarsa_greedy_policy
from rl.dqn import dqn, run_dqn_experiments, greedy_policy as dqn_greedy_policy
import os

'''
    learn.py
    Entry point for rl learning algorithms
    NOTE: 
        Theoretical max reward of 3500 when number enemies = 5 (excluding tower level ups) per episode
        max_reward = 10 * (\sum_{i=1}^{n} (num_enemies + 2 * i - 1)) + 2000
'''

def run_q_learning():
    # env = TowerDefenseEnv(render_mode="human", render_rate=100) 
    env = TowerDefenseEnv(render_mode=None)
    
    # Approximate Q values
    Q = q_learning(env, 
                   episodes=5000, 
                   alpha=0.5, 
                   gamma=0.9, 
                   epsilon_start=0.9, 
                   epsilon_end=0.05,
                   epsilon_decay_steps=1500,
                   log=True)
    env.close()

    env = TowerDefenseEnv(render_mode="human", render_rate=200) 
    # evaluate_policy(env, Q, episodes=5, sleep=0.5)
    evaluate_policy(env, Q)
    env.close()

def run_a2c_learning():
    print(f"[A2C LEARNING] Starting A2C Learning")

    # TRAINING 
    env = TowerDefenseEnv(render_mode=None) 
    featurizer = TileCoder(env, num_tilings=16, tiles_per_dim=8, max_size=8192)

    start_time = time.time()
    Theta, w, eval_returns = a2c(env, featurizer, evaluate_policy_fa)
    print(f"[A2C TRAINING] Finished in {time.time() - start_time:.3f} seconds")
    env.close()
    
    # VISUALIZE Learned Policy 
    env = TowerDefenseEnv(render_mode="human", render_rate=50) 
    render_env(env, featurizer, Theta, softmaxPolicy)
    env.close()

    # TEST Learned Policy 
    env = TowerDefenseEnv(render_mode=None) 
    # featurizer = RbfFeaturizer(env, 100)
    featurizer = TileCoder(env, num_tilings=16, tiles_per_dim=8, max_size=8192)
    img_dest_path = "/home/andrew/cu/comp4010-project/project_code/rl/results/"
    file_name = "a2c_learn"
    start_time = time.time()
    results = run_a2c_experiments(env=env, featurizer=featurizer, eval_func=evaluate_policy_fa, img_dest_path=img_dest_path, file_name=file_name)  
    print(f"[A2C TESTING] Finished in {time.time() - start_time:.3f} seconds")
    env.close()

    return

def run_sarsa_learning():
    print(f"[SARSA LEARNING] Starting SARSA Learning")
    num_enemies = 7
    # # TRAINING 
    # env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    # start_time = time.time()
    # q_net, eval_returns = sarsa(env=env, eval_func=evaluate_policy_nn)
    # print(f"[SARSA TRAINING] Finished in {time.time() - start_time:.3f} seconds")
    # env.close()
    
    # # VISUALIZE Learned Policy 
    # env = TowerDefenseEnv(render_mode="human", render_rate=50, num_enemies=num_enemies) 
    # def render_env(env: gym.Env, nn, policy_func):
    #     observation = env.reset()[0]
    #     while True:
    #         env.render()
    #         action = policy_func(observation, nn)
    #         observation, reward, terminated, truncated, info = env.step(action)
    #         if terminated or truncated:
    #             print(f"Terminated or truncated")
    #             break

    #     env.close()
    #     return
    # render_env(env, q_net, sarsa_greedy_policy)

    # TEST Learned Policy 
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    img_dest_path = "/home/andrew/cu/comp4010-project/project_code/rl/results/"
    file_name = "sarsa_learn"
    start_time = time.time()
    results = run_sarsa_experiments(env=env, eval_func=evaluate_policy_nn, img_dest_path=img_dest_path, file_name=file_name)  
    print(f"[SARSA TESTING] Finished in {time.time() - start_time:.3f} seconds")
    env.close()


def run_dqn_learning():
    num_enemies = 7
    
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    
    # Use current directory for results
    img_dest_path = os.path.dirname(os.path.abspath(__file__)) + "/results"
    file_name = "dqn_learn"
    
    start_time = time.time()
    results = run_dqn_experiments(
        env=env, 
        eval_func=evaluate_policy_nn, 
        img_dest_path=img_dest_path, 
        file_name=file_name
    )  
    env.close()
    
    return results

