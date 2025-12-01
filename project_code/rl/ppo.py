import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

"""
    ppo.py
    PPO implementation using Stable-Baselines3 library.
"""


class NormalizedEnvWrapper(gym.Wrapper):
    """Wrapper that applies min-max normalization to observations (same as Q-Learning/SARSA/DQN)"""
    
    def __init__(self, env):
        super().__init__(env)
        # Build max values vector for normalization
        state_dim = env.observation_space.shape[0]
        self.max_vals = np.ones(state_dim, dtype=np.float32)
        for i in range(100):  # 10x10 grid
            base = i * 8
            self.max_vals[base + 0] = 2.0     # tower id
            self.max_vals[base + 1] = 5.0     # tower level
            self.max_vals[base + 2] = 10.0    # num enemies
            self.max_vals[base + 3] = 50.0    # avg enemy health
            self.max_vals[base + 4] = 50.0    # max enemy health
            self.max_vals[base + 5] = 1.0     # path
            self.max_vals[base + 6] = 1.0     # base indicator
            self.max_vals[base + 7] = 40.0    # base health
        self.max_vals[-1] = 10000.0  # budget
        
        # Update observation space to normalized range
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
    
    def _normalize(self, obs):
        return np.clip(obs / self.max_vals, 0.0, 1.0).astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info


class EvalCallback(BaseCallback):
    """Callback for evaluating the policy during training"""
    
    def __init__(self, eval_env, eval_func, evaluate_every=20, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_func = eval_func
        self.evaluate_every = evaluate_every
        self.n_eval_episodes = n_eval_episodes
        self.eval_returns = []
        self.episode_count = 0
        self.last_eval_episode = 0
    
    def _on_step(self) -> bool:
        # Count episodes from dones
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self.episode_count += 1
        
        # Evaluate every N episodes
        if self.episode_count >= self.last_eval_episode + self.evaluate_every:
            self.last_eval_episode = self.episode_count
            
            # Evaluate using the provided eval function
            def policy_fn(obs):
                # Normalize obs before prediction
                if hasattr(self.model.env.envs[0], 'max_vals'):
                    normalized = np.clip(obs / self.model.env.envs[0].max_vals, 0.0, 1.0).astype(np.float32)
                else:
                    normalized = obs
                action, _ = self.model.predict(normalized, deterministic=True)
                return action
            
            # Run evaluation on unwrapped env
            if hasattr(self.eval_env.envs[0], 'env'):
                unwrapped_env = self.eval_env.envs[0].env
            else:
                unwrapped_env = self.eval_env.envs[0]
                
            avg_return = self.eval_func(
                unwrapped_env,
                policy_func=policy_fn,
                n_runs=self.n_eval_episodes
            )
            self.eval_returns.append(avg_return)
            
            if self.verbose:
                print(f"[PPO] Episode {self.episode_count}: Avg Return = {avg_return:.2f}")
        
        return True


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
        normalize=True,
        verbose=1):

    # Wrap environment with normalization and vectorization
    if normalize:
        wrapped_env = NormalizedEnvWrapper(env)
        vec_env = DummyVecEnv([lambda: wrapped_env])
        eval_env = DummyVecEnv([lambda: NormalizedEnvWrapper(env)])
    else:
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
        verbose=0,  # Reduce SB3 verbosity
        seed=101263813
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_func=eval_func,
        evaluate_every=evaluate_every,
        verbose=verbose
    )
    
    # Estimate timesteps (average ~50 steps per episode in this env)
    total_timesteps = max_episodes * 50
    
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
        
        max_len = max(len(run) for run in eval_returns_runs) if eval_returns_runs else 0
        if max_len == 0:
            return np.array([])
            
        padded_runs = np.zeros((n_runs, max_len))
        for i, run in enumerate(eval_returns_runs):
            padded_runs[i, :len(run)] = run
            if len(run) < max_len:
                padded_runs[i, len(run):] = run[-1] if run else 0
        
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
        if len(results[i]) > 0:
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
        max_len = max(len(run) for run in eval_returns_runs) if eval_returns_runs else 0
        if max_len == 0:
            return np.array([])
            
        padded_runs = np.zeros((n_runs, max_len))
        for i, run in enumerate(eval_returns_runs):
            padded_runs[i, :len(run)] = run
            if len(run) < max_len:
                padded_runs[i, len(run):] = run[-1] if run else 0
        
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
        if len(results[i]) > 0:
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
