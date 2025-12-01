
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

def run_a2c_sb3_experiments(env, img_dest_path="", file_name=""):
    """
    Runs experiments for A2C using Stable-Baselines3
    Args:
        env: A gym environment
        eval_func: a function to evaluate performance of the current policy
        img_dest_path: path to save results images
        file_name: filename prefix for saved plots
    Returns:
        results: array of evaluated returns
    """

    n_runs = 1
    max_episodes = 1000
    evaluate_every = 50
    n_eval = max_episodes // evaluate_every  # number of evaluations during training

    actor_lr = 5e-5
    critic_lr_list = [5e-5]  # Example critic step sizes
    results = np.zeros([len(critic_lr_list), n_eval])

    def repeat_experiments(actor_lr, critic_lr):
        eval_returns_step_sizes = np.zeros([n_runs, n_eval])
        for r in range(n_runs):

            # Wrap environment in a vectorized wrapper
            vec_env = DummyVecEnv([lambda: env])

            # Custom policy kwargs with separate learning rates
            policy_kwargs = dict(net_arch=[256, 256])  # MLP; can adjust
            model = A2C(
                "MlpPolicy",
                vec_env,
                learning_rate=actor_lr,
                gamma=0.99,
                verbose=0,
                policy_kwargs=policy_kwargs,
            )

            eval_returns = []
            obs = vec_env.reset()
            for episode in range(1, max_episodes + 1):
                model.learn(total_timesteps=1, reset_num_timesteps=False)  # Step through environment

                # Evaluate every `evaluate_every` episodes
                if episode % evaluate_every == 0:
                    avg_return, std_return = evaluate_policy(model, env)
                    eval_returns.append(avg_return)

            eval_returns_step_sizes[r] = eval_returns
            vec_env.close()
        return np.mean(eval_returns_step_sizes, axis=0)

    np.random.seed(101210291)

    # Repeat experiments for different critic learning rates
    for i, critic_lr in enumerate(critic_lr_list):
        results[i] = repeat_experiments(actor_lr=actor_lr, critic_lr=critic_lr)

    # Plot results
    plt.figure()
    x = np.arange(1, n_eval + 1)
    for i, critic_lr in enumerate(critic_lr_list):
        plt.plot(x, results[i], label=f"Critic step size = {critic_lr}")

    plt.xlabel(f"Evaluation Number (Every {evaluate_every} Episodes)")
    plt.ylabel("Average Evaluated Return")
    plt.title(f"A2C with SB3 (Actor LR = {actor_lr})\nAverage Return over {n_runs} run(s) ({max_episodes} episodes per run)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{img_dest_path}/{file_name}_{n_runs}runs_{max_episodes}episodes.png")
    plt.clf()

    return results


import numpy as np

def evaluate_policy(model, env, n_eval_episodes=5, render=False):
    """
    Evaluate an SB3 model on a Gym environment (works with Gym >=0.26 and VecEnv).
    
    Args:
        model: Stable-Baselines3 model (A2C, PPO, etc.)
        env: Gym environment (can be VecEnv or standard Gym)
        n_eval_episodes: Number of episodes to run for evaluation
        render: Whether to render the environment during evaluation
        
    Returns:
        avg_return: Average total reward
        std_return: Standard deviation of total rewards
    """
    returns = []

    for _ in range(n_eval_episodes):
        obs = env.reset()

        # Unpack tuple if using Gym >=0.26
        if isinstance(obs, tuple):
            obs = obs[0]

        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated or truncated:
            # Ensure obs is a numpy array
            obs_array = np.array(obs)

            action, _ = model.predict(obs_array, deterministic=True)
            obs, reward, terminated, truncated, _= env.step(action)

            # Unpack again if env.step() returns a tuple
            if isinstance(obs, tuple):
                obs = obs[0]

            total_reward += reward

            if render:
                env.render()

        returns.append(total_reward)

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    return avg_return, std_return
