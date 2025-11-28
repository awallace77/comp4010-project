from envs.tower_defense_env import TowerDefenseEnv
import time
import numpy as np
import gymnasium as gym
from rl.utils import RbfFeaturizer, evaluate, render_env

# Algo imports
from rl.a2c import ActorCritic, softmaxPolicy, runACExperiments
from rl.qlearn import q_learning, evaluate_policy

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

    # TRAINING 
    # env = TowerDefenseEnv(render_mode=None, render_rate=200) 
    # featurizer = RbfFeaturizer(env, 100)
    # start_time = time.time()
    # Theta, w, eval_returns = ActorCritic(env, featurizer, evaluate)
    # print(f'[A2C TRAINING] Finished in {time.time() - start_time:.3f} seconds')
    # env.close()
    
    # VISUALIZE Learned Policy 
    # env = TowerDefenseEnv(render_mode="human", render_rate=200) 
    # render_env(env, featurizer, Theta, softmaxPolicy)
    # env.close()

    # TEST Learned Policy 
    env = TowerDefenseEnv(render_mode=None) 
    featurizer = RbfFeaturizer(env, 100)
    img_dest_path = "/home/andrew/cu/comp4010-project/project_code/rl/results/"
    file_name = "a2c_learn"
    start_time = time.time()
    results = runACExperiments(env=env, featurizer=featurizer, eval_func=evaluate, img_dest_path=img_dest_path, file_name=file_name)  
    print(f'[A2C TESTING] Finished in {time.time() - start_time:.3f} seconds')
    env.close()

    return