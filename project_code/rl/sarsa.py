import numpy as np
import torch
import torch.optim as optim
from rl.nnetworks.qnetwork import QNetwork
from matplotlib import pyplot as plt
import math
from rl.utils import log
"""
    SARSA w Q-Network Function Approximation 
"""

def sarsa(
        env, 
        eval_func, 
        gamma=0.99, 
        learning_rate=0.0005, 
        epsilon_start=1.0, 
        epsilon_min=0.05, 
        epsilon_decay=0.995, 
        max_episode=10000,
        evaluate_every=50):

    # Get environment dimensions
    obs_dim = env._get_obs().shape[0]
    num_actions = env.action_space.n

    # Initialize Q-network
    q_net = QNetwork(input_dim=obs_dim, output_dim=num_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.SmoothL1Loss() # Huber loss

    eval_returns = []
    epsilon = epsilon_start
    for episode in range(1, max_episode + 1):
        
        # Initial state
        state = env.reset()[0]
        action = e_greedy(env, state, q_net, epsilon)
        done = False
        
        while not done:
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Select next action using epsilon-greedy
            next_action = e_greedy(env, next_state, q_net, epsilon)

            # Normalize for nn
            state_tensor = normalize_state(state)
            next_state_tensor = normalize_state(next_state)

            q_values = q_net(state_tensor.unsqueeze(0))
            q_current = q_values[0, action]

            # Compute TD target
            with torch.no_grad():
                if done:
                    td_target_val = float(reward)
                else:
                    q_next = q_net(next_state_tensor.unsqueeze(0))
                    q_next_a = float(q_next[0, next_action].item())
                    td_target_val = float(reward) + gamma * q_next_a

            # Convert to tensor
            td_target = torch.tensor(td_target_val, dtype=q_current.dtype, device=q_current.device)
 
            # Loss
            loss = loss_fn(q_current, td_target)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0) # Prevent exploding gradients
            optimizer.step()

            # Move to next step
            state = next_state
            action = next_action

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % evaluate_every == 0:
            eval_return = eval_func(env, lambda state: greedy_policy(state, q_net))
            eval_returns.append(eval_return)
        
        if episode % math.floor(max_episode / 10) == 0:
            log("SARSA", f"Episode {episode} epsilon = {epsilon:.4f}")

    return q_net, eval_returns

def run_sarsa_experiments(env, eval_func, img_dest_path="", file_name=""):    
    """
        Runs experiments for SARSA
        Args:
            env: A gym environment
            eval_func: an evaluation function that evaluates the performance of the intermediate policy during training
            img_dest_path: absolute path to save image of results to 
            file_name: the name of the resulting image
        Returns:
            results: the results of the experiments
    """

    def repeatExperiments(learning_rate):
        eval_returns_step_sizes = np.zeros([n_runs, n_eval])
        for r in range(n_runs):
            q_net, eval_returns = sarsa(
                env,
                eval_func=eval_func,
                gamma=0.99,
                learning_rate=learning_rate,
                max_episode=max_episodes,
                evaluate_every=evaluate_every)

            eval_returns_step_sizes[r] = eval_returns
        
        # Compute and return the average evaluated returns over runs
        avg_eval_returns = np.mean(eval_returns_step_sizes, axis=0) 
        return avg_eval_returns

    # n_runs = 100
    n_runs = 1
    max_episodes = 10000
    evaluate_every = 100
    n_eval = max_episodes // evaluate_every # num of evaluation during training 

    def save_plt():
        plt.xlabel(f"Evaluation Number (Every {evaluate_every} Episodes)")
        plt.ylabel("Average Evaluated Return")
        title = f"SARSA with Q-Network Function Approximation"
        title = title + f"\nAverage Return over {n_runs} runs ({max_episodes} episodes per run)"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        path = f"{img_dest_path}/{file_name}_{n_runs}runs_{max_episodes}episodes.png"
        plt.savefig(path)
        plt.clf()

    # learning_rate_list = [0.05, 0.01, 0.005, 0.001]
    # learning_rate_list = [0.001, 0.0005, 0.0001, 0.00001]
    learning_rate_list = [0.0001]
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

# build max values vector once at module load
_max_vals = None

def _get_max_vals(size):
    """get or create max values vector"""
    global _max_vals
    if _max_vals is None or len(_max_vals) != size:
        _max_vals = torch.ones(size)
        for i in range(100):  # 10x10 grid
            base = i * 8
            _max_vals[base + 0] = 2.0     # tower id
            _max_vals[base + 1] = 5.0     # tower level
            _max_vals[base + 2] = 10.0    # num enemies
            _max_vals[base + 3] = 50.0    # avg enemy health
            _max_vals[base + 4] = 50.0    # max enemy health
            _max_vals[base + 5] = 1.0     # path
            _max_vals[base + 6] = 1.0     # base indicator
            _max_vals[base + 7] = 40.0    # base health
        _max_vals[-1] = 10000.0  # budget
    return _max_vals

def normalize_state(state):
    """min-max normalization per feature"""
    s = torch.tensor(state, dtype=torch.float32)
    max_vals = _get_max_vals(len(s))
    return torch.clamp(s / max_vals, 0.0, 1.0)

def greedy_policy(state, q_net):
    if not isinstance(state, torch.Tensor):
        s = normalize_state(state)
    else:
        s = state / (state.norm() + 1e-8)
    with torch.no_grad():
        q_vals = q_net(s.unsqueeze(0))
        return int(torch.argmax(q_vals, dim=1).item())

def e_greedy(env, state, q_net, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return greedy_policy(state, q_net)
