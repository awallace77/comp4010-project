import gymnasium as gym
import numpy as np
from sb3_contrib import QRDQN
from matplotlib import pyplot as plt

def qrdqn(
        env, 
        eval_func, 
        learning_rate=5e-5, 
        total_timesteps=100_000, 
        evaluate_every=5000, 
        render=False):
    policy_kwargs = dict(n_quantiles=50)
    model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=learning_rate, verbose=0)
    
    eval_returns = []
    timesteps_done = 0
    
    while timesteps_done < total_timesteps:
        model.learn(total_timesteps=evaluate_every)
        timesteps_done += evaluate_every
        
        # Evaluate current policy
        mean_return = eval_func(env, lambda state: model.predict(state, deterministic=True)[0])
        eval_returns.append(mean_return)
    
    if render:
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
    
    return model, eval_returns

def run_qrdqn_experiments(env, img_dest_path="", file_name=""):    
    """
        Runs experiments for QRDQN
        Args:
            env: A gym environment
            eval_func: an evaluation function that evaluates the performance of the intermediate policy during training
            img_dest_path: absolute path to save image of results to 
            file_name: the name of the resulting image
        Returns:
            results: the results of the experiments
    """

    def eval_func(env, policy_fn, n_episodes=5):
        returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            returns.append(total_reward)
        return np.mean(returns)

    def repeatExperiments(learning_rate):
        eval_returns_step_sizes = np.zeros((n_runs, n_eval))
        for r in range(n_runs):
            model, eval_returns = qrdqn(
                env,
                eval_func=eval_func,
                learning_rate=learning_rate,
                total_timesteps=total_timesteps,
                evaluate_every=evaluate_every)

            eval_returns_step_sizes[r] = eval_returns
        
        # Compute and return the average evaluated returns over runs
        avg_eval_returns = np.mean(eval_returns_step_sizes, axis=0) 
        return avg_eval_returns

    # n_runs = 100
    total_timesteps=100000
    evaluate_every=100
    n_runs=2
    n_eval = total_timesteps // evaluate_every

    def save_plt():
        plt.xlabel(f"Evaluation Number (Every {evaluate_every} Steps)")
        plt.ylabel("Average Evaluated Return")
        title = f"QRDQN Learning"
        title = title + f"\nAverage Return over {n_runs} runs ({total_timesteps} total time steps per run)"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        path = f"{img_dest_path}/{file_name}_{n_runs}runs_{total_timesteps}timesteps.png"
        plt.savefig(path)
        plt.clf()

    # learning_rate_list = [0.05, 0.01, 0.005, 0.001]
    learning_rate_list = [0.001, 0.0005, 0.0001]
    results = np.zeros((len(learning_rate_list), n_eval))
    np.random.seed(101210291)

    for i, learning_rate in enumerate(learning_rate_list):
        results[i] = repeatExperiments(learning_rate=learning_rate)

    plt.figure()
    x = np.arange(1, n_eval + 1) 
    for i, learning_rate in enumerate(learning_rate_list):
        plt.plot(x, results[i], label=f"Learning Rate = {learning_rate}")
    save_plt()
    
    return results 