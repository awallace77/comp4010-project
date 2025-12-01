"""
    run_evaluation.py
    Full evaluation script for all RL algorithms
    COMP4010 - Group 12
"""
import os
import time
import numpy as np

from envs.tower_defense_env import TowerDefenseEnv
from rl.evaluation import AlgorithmComparator

from rl.qlearning import qlearning
from rl.sarsa import sarsa
from rl.a2c import a2c
from rl.dqn import dqn
from rl.featurizers.tile_coder_featurizer import TileCoder
from rl.utils import evaluate_policy_fa, evaluate_policy_nn


# config
NUM_ENEMIES = 7
EPISODES = 5000
EVALUATE_EVERY = 50
OUTPUT_DIR = "evaluation_results"


def train_qlearning(env):
    """train q-learning and return eval results"""
    print("\n" + "="*50)
    print("Training Q-Learning...")
    print("="*50)
    
    start = time.time()
    q_net, eval_returns = qlearning(
        env=env,
        eval_func=evaluate_policy_nn,
        max_episode=EPISODES,
        evaluate_every=EVALUATE_EVERY
    )
    print(f"Training done in {time.time() - start:.1f}s")
    
    return eval_returns


def train_sarsa(env):
    """train sarsa and return eval results"""
    print("\n" + "="*50)
    print("Training SARSA...")
    print("="*50)
    
    start = time.time()
    q_net, eval_returns = sarsa(
        env=env,
        eval_func=evaluate_policy_nn,
        max_episode=EPISODES,
        evaluate_every=EVALUATE_EVERY
    )
    print(f"Training done in {time.time() - start:.1f}s")
    
    return eval_returns


def train_dqn(env):
    """train dqn and return eval results"""
    print("\n" + "="*50)
    print("Training DQN...")
    print("="*50)
    
    start = time.time()
    q_net, eval_returns = dqn(
        env=env,
        eval_func=evaluate_policy_nn,
        max_episode=EPISODES,
        evaluate_every=EVALUATE_EVERY
    )
    print(f"Training done in {time.time() - start:.1f}s")
    
    return eval_returns


def train_a2c(env):
    """train a2c and return eval results"""
    print("\n" + "="*50)
    print("Training A2C...")
    print("="*50)
    
    featurizer = TileCoder(env, num_tilings=16, tiles_per_dim=8, max_size=8192)
    
    start = time.time()
    Theta, w, eval_returns = a2c(
        env=env,
        featurizer=featurizer,
        eval_func=evaluate_policy_fa,
        max_episodes=EPISODES,
        evaluate_every=EVALUATE_EVERY
    )
    print(f"Training done in {time.time() - start:.1f}s")
    
    return eval_returns


def main():
    """main evaluation pipeline"""
    print("="*60)
    print("  TOWER DEFENSE RL - FULL EVALUATION")
    print("  COMP4010 Group 12")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Episodes: {EPISODES}")
    print(f"  Evaluate every: {EVALUATE_EVERY}")
    print(f"  Num enemies: {NUM_ENEMIES}")
    print(f"  Output dir: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    env = TowerDefenseEnv(render_mode=None, num_enemies=NUM_ENEMIES)
    comparator = AlgorithmComparator()
    
    # train and collect results
    try:
        eval_returns = train_qlearning(env)
        comparator.add_result("Q-Learning", eval_returns, EVALUATE_EVERY)
    except Exception as e:
        print(f"Q-Learning failed: {e}")
    
    try:
        eval_returns = train_sarsa(env)
        comparator.add_result("SARSA", eval_returns, EVALUATE_EVERY)
    except Exception as e:
        print(f"SARSA failed: {e}")
    
    try:
        eval_returns = train_dqn(env)
        comparator.add_result("DQN", eval_returns, EVALUATE_EVERY)
    except Exception as e:
        print(f"DQN failed: {e}")
    
    try:
        eval_returns = train_a2c(env)
        comparator.add_result("A2C", eval_returns, EVALUATE_EVERY)
    except Exception as e:
        print(f"A2C failed: {e}")
    
    # final comparison
    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    comparator.print_comparison()
    comparator.save_comparison(OUTPUT_DIR)
    
    env.close()
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
