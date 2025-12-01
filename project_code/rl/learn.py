import time
import numpy as np
import gymnasium as gym
import os

from rl.utils import evaluate_policy_fa, evaluate_policy_nn, render_env
from envs.tower_defense_env import TowerDefenseEnv
from rl.utils import evaluate_policy_fa, evaluate_policy_nn, render_env, log
from rl.featurizers.tile_coder_featurizer import TileCoder

# Algo imports
from rl.qlearning import qlearning, run_qlearning_experiments, greedy_policy as qlearning_greedy_policy, Featurizer
from rl.a2c import a2c, softmaxPolicy, run_a2c_experiments 
from rl.sarsa import sarsa, run_sarsa_experiments, greedy_policy as sarsa_greedy_policy
from rl.ppo import ppo, run_ppo_experiments, run_ppo_lr_experiments, greedy_policy
from rl.qrdqn import run_qrdqn_experiments

"""
    learn.py
    Entry point for rl learning algorithms
    NOTE: 
        Theoretical max reward of 3500 when number enemies = 5 (excluding tower level ups) per episode
        max_reward = 10 * (sum_{i=1}^{n} (num_enemies + 2 * i - 1)) + 2000
"""
from rl.dqn import dqn, run_dqn_experiments, greedy_policy as dqn_greedy_policy
import os

'''
    learn.py
    Entry point for rl learning algorithms
    NOTE: 
        Theoretical max reward of 3500 when number enemies = 5 (excluding tower level up rewards) per episode
        max_reward = 10 * (\sum_{i=1}^{10} (num_enemies + 2 * i - 1)) + 2000
'''

# NOTE : Please update image destination path to your dir
num_waves = 10
num_enemies = 7
img_dest_path = "/home/andrew/cu/comp4010-project/project_code/rl/results/"

def approx_max_reward(num_waves, num_enemies):
    max_reward = num_waves * np.sum([num_enemies + 2 * i -1 for i in range(1, num_waves + 1)]) + 200 * num_waves
    return max_reward

log("INFO", f"Approximate max reward: {approx_max_reward(num_waves, num_enemies)}")

def run_qlearning():
    level = "Q-LEARNING"
    log(level, "Starting Q-Learning")
    num_enemies = 3
    
    # TRAINING 
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    featurizer = Featurizer(env.observation_space.shape[0])
    start_time = time.time()
    q_net, eval_returns = qlearning(env=env, eval_func=evaluate_policy_nn, featurizer=featurizer, max_episode=1000, learning_rate=0.001)
    log(level, f"Training finished in {time.time() - start_time:.3f} seconds")
    env.close()
    
    # VISUALIZE
    env = TowerDefenseEnv(render_mode="human", render_rate=50, num_enemies=num_enemies) 
    start_time = time.time()
    def render_env(env: gym.Env, nn, policy_func):
        observation = env.reset()[0]
        while True:
            env.render()
            action = policy_func(observation, nn, featurizer)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        env.close()
        return
    render_env(env, q_net, qlearning_greedy_policy)
    log(level, f"Visualization finished in {time.time() - start_time:.3f} seconds")

    # TESTING
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 

    file_name = "qlearning"
    start_time = time.time()
    results = run_qlearning_experiments(
        env=env, 
        eval_func=evaluate_policy_nn, 
        featurizer=featurizer,
        img_dest_path=img_dest_path, 
        file_name=file_name)
    log(level, f"Testing finished in {time.time() - start_time:.3f} seconds")

    env.close()


def run_a2c_learning():
    level = "A2C"
    log(level, f"Starting A2C Learning")

    # TRAINING 
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    featurizer = TileCoder(env, num_tilings=16, tiles_per_dim=8, max_size=8192)
    start_time = time.time()
    Theta, w, eval_returns = a2c(env, featurizer, evaluate_policy_fa)
    log(level, f"Training finished in {time.time() - start_time:.3f} seconds")
    env.close()
    
    # VISUALIZE
    start_time = time.time()
    env = TowerDefenseEnv(render_mode="human", render_rate=50, num_enemies=num_enemies) 
    render_env(env, featurizer, Theta, softmaxPolicy)
    log(level, f"Visualization finished in {time.time() - start_time:.3f} seconds")
    env.close()

    # EVALUATE 
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    featurizer = TileCoder(env, num_tilings=16, tiles_per_dim=8, max_size=8192)
    file_name = "a2c"
    start_time = time.time()
    results = run_a2c_experiments(env=env, featurizer=featurizer, eval_func=evaluate_policy_fa, img_dest_path=img_dest_path, file_name=file_name)  
    log(level, f"Testing finished in {time.time() - start_time:.3f} seconds")
    env.close()

    return

def run_sarsa_learning():
    level = "SARSA"
    log(level, f"Starting SARSA Learning")
    
    # TRAINING 
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    start_time = time.time()
    q_net, eval_returns = sarsa(env=env, eval_func=evaluate_policy_nn, max_episode=200)
    log(level, f"Training finished in {time.time() - start_time:.3f} seconds")
    env.close()
    
    # VISUALIZE Learned Policy 
    env = TowerDefenseEnv(render_mode="human", render_rate=50, num_enemies=num_enemies) 
    def render_env(env: gym.Env, nn, policy_func):
        observation = env.reset()[0]
        while True:
            env.render()
            action = policy_func(observation, nn)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        env.close()
        return
    start_time = time.time()
    render_env(env, q_net, sarsa_greedy_policy)
    log(level, f"Visualization finished in {time.time() - start_time:.3f} seconds")
    env.close()

    # TEST Learned Policy 
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 
    file_name = "sarsa"
    start_time = time.time()
    results = run_sarsa_experiments(
        env=env, 
        eval_func=evaluate_policy_nn, 
        img_dest_path=img_dest_path, 
        file_name=file_name)
    log(level, f"Testing finished in {time.time() - start_time:.3f} seconds")
    env.close()


def run_ppo_learning():
    num_enemies = 5  

    # TRAINING
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies)
    model, eval_returns = ppo(
        env=env,
        eval_func=evaluate_policy_nn,
        learning_rate=3e-4,
        clip_range=0.2,
        max_episodes=30,       
        evaluate_every=10,
        verbose=1
    )
    env.close()

    # TEST Learned Policy with experiments
    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies)
    img_dest_path = os.path.join(os.path.dirname(__file__), "results")
    file_name = "ppo_learn"
    results = run_ppo_lr_experiments(
        env=env,
        eval_func=evaluate_policy_nn,
        img_dest_path=img_dest_path,
        file_name=file_name
    )
    env.close()

    return


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

def run_qrdqn_learning():
    num_enemies = 5

    env = TowerDefenseEnv(render_mode=None, num_enemies=num_enemies) 

    # Use current directory for results
    img_dest_path = os.path.dirname(os.path.abspath(__file__)) + "/results"
    file_name = "qrdqn_learn"

    start_time = time.time()
    results = run_qrdqn_experiments(
        env=env, 
        img_dest_path=img_dest_path, 
        file_name=file_name
    )  
    log("QRDQN", f"Testing finished in {time.time() - start_time:.3f} seconds")
    env.close()

    return results