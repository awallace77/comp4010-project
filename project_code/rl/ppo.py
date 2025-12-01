import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

"""
    ppo.py
    PPO implementation using Stable-Baselines3 library.
"""

def ppo(
        env,
        eval_func,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        max_episodes=1000,
        evaluate_every=20,
        policy_kwargs=None,
        verbose=1):

    # Wrap environment for vectorized training
    vec_env = DummyVecEnv([lambda: env])
    eval_env = DummyVecEnv([lambda: env])
    
    # Default policy kwargs with MLP architecture
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=[128, 128] 
        )
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=101263813
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env=eval_env,
        # eval_func=eval_func,
        eval_freq=evaluate_every,
        verbose=verbose
    )
    
    total_timesteps = max_episodes * 1000   
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=False  
    )
    
    return model, eval_callback.eval_returns


def greedy_policy(state, model):
    # Returns the best action for given state using SB3 model.
    action, _ = model.predict(state, deterministic=True)
    return action


def stochastic_policy(state, model):
    # Sample action from policy distribution using SB3 model.
    action, _ = model.predict(state, deterministic=False)
    return action


def run_ppo_experiments(env, eval_func, img_dest_path="", file_name=""):
    # Runs experiments for PPO with learning rate and clip range 
 
    def repeat_experiments(learning_rate, clip_range):
        eval_returns_runs = []
        for r in range(n_runs):
            print(f"[PPO] Run {r+1}/{n_runs} (LR={learning_rate}, clip={clip_range})")
            model, eval_returns = ppo(
                env,
                eval_func=eval_func,
                learning_rate=learning_rate,
                clip_range=clip_range,
                max_episodes=max_episodes,
                evaluate_every=evaluate_every,
                verbose=0
            )
            eval_returns_runs.append(eval_returns)
        
        max_len = max(len(run) for run in eval_returns_runs)
        padded_runs = np.zeros((n_runs, max_len))
        for i, run in enumerate(eval_returns_runs):
            padded_runs[i, :len(run)] = run
            if len(run) < max_len:
                padded_runs[i, len(run):] = run[-1]
        
        # Compute average
        avg_eval_returns = np.mean(padded_runs, axis=0)
        return avg_eval_returns
    
    n_runs = 5
    max_episodes = 500
    evaluate_every = 20
    
    # Hyperparameter variations
    learning_rate = 3e-4
    clip_range_list = [0.1, 0.2, 0.3]
    results = []
    
    np.random.seed(101263813)
    
    # Repeat experiments with different clip ranges
    for clip_range in clip_range_list:
        avg_returns = repeat_experiments(
            learning_rate=learning_rate,
            clip_range=clip_range
        )
        results.append(avg_returns)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for i, clip_range in enumerate(clip_range_list):
        x = np.arange(1, len(results[i]) + 1) * evaluate_every
        plt.plot(x, results[i], label=f"Clip ε = {clip_range}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Average Evaluated Return")
    plt.title("PPO Performance (Learning Rate = 3e-4)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if img_dest_path and file_name:
        plt.savefig(f"{img_dest_path}/{file_name}.png", dpi=150, bbox_inches='tight')
    plt.clf()
    
    return results


def run_ppo_lr_experiments(env, eval_func, img_dest_path="", file_name=""):  
    # Runs experiments for PPO comparing different learning rates.
     
    def repeat_experiments(learning_rate):
        eval_returns_runs = []
        for r in range(n_runs):
            print(f"[PPO] Run {r+1}/{n_runs} (LR={learning_rate})")
            model, eval_returns = ppo(
                env,
                eval_func=eval_func,
                learning_rate=learning_rate,
                clip_range=0.2,
                max_episodes=max_episodes,
                evaluate_every=evaluate_every,
                verbose=0
            )
            eval_returns_runs.append(eval_returns)
        
        # Pad eval_returns to same length
        max_len = max(len(run) for run in eval_returns_runs)
        padded_runs = np.zeros((n_runs, max_len))
        for i, run in enumerate(eval_returns_runs):
            padded_runs[i, :len(run)] = run
            if len(run) < max_len:
                padded_runs[i, len(run):] = run[-1]
        
        avg_eval_returns = np.mean(padded_runs, axis=0)
        return avg_eval_returns
    
    n_runs = 5
    max_episodes = 500
    evaluate_every = 20
    
    # Learning rate variations
    learning_rate_list = [1e-3, 3e-4, 1e-4, 3e-5]
    results = []
    
    np.random.seed(101210291)
    
    for lr in learning_rate_list:
        avg_returns = repeat_experiments(learning_rate=lr)
        results.append(avg_returns)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for i, lr in enumerate(learning_rate_list):
        x = np.arange(1, len(results[i]) + 1) * evaluate_every
        plt.plot(x, results[i], label=f"LR = {lr}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Average Evaluated Return")
    plt.title("PPO (Stable-Baselines3) Performance (Clip ε = 0.2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if img_dest_path and file_name:
        plt.savefig(f"{img_dest_path}/{file_name}.png", dpi=150, bbox_inches='tight')
    plt.clf()
    
    return results


