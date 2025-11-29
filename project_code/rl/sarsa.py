import numpy as np
import torch
import torch.optim as optim
from rl.nnetworks.qnetwork import QNetwork
from matplotlib import pyplot as plt

def sarsa(
        env, 
        eval_func, 
        gamma=0.99, 
        learning_rate=0.01, 
        epsilon_start=0.995, 
        epsilon_min=0.05, 
        epsilon_decay=0.005, 
        max_episode=1000,
        evaluate_every=20):

    # Get environment dimensions
    obs_dim = env._get_obs().shape[0]  # flattened obs + budget
    num_actions = env.action_space.n

    # Initialize Q-network
    q_net = QNetwork(input_dim=obs_dim, output_dim=num_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    eval_returns = []
    for episode in range(1, max_episode + 1):
        
        # Epsilon decay for early exploration
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(-epsilon_decay * episode)

        state = env.reset()[0]
        action = e_greedy(env, state, q_net, epsilon)
        done = False
        
        while not done:
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Select next action using epsilon-greedy
            next_action = e_greedy(env, next_state, q_net, epsilon)

            # Normalize
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            next_state_tensor /= np.linalg.norm(next_state_tensor)

            # Compute TD target: r + gamma * Q(next_state, next_action)
            q_values_next = q_net(next_state_tensor.unsqueeze(0))
            td_target = reward + gamma * q_values_next[0, next_action].item() * (not done)

            # Current Q value
            q_values = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            q_current = q_values[0, action]

            # Loss
            td_target_tensor = torch.tensor(td_target, dtype=torch.float32)
            loss = loss_fn(q_current, td_target_tensor)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

            # Move to next step
            state = next_state
            action = next_action

        if episode % evaluate_every == 0:
            eval_return = eval_func(env, lambda state: greedy_policy(state, q_net))
            eval_returns.append(eval_return)

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
    n_runs = 10
    max_episodes = 2000
    evaluate_every = 20
    n_eval = max_episodes // evaluate_every # num of evaluation during training

    learning_rate_list = [0.05, 0.01, 0.005, 0.001]
    results = np.zeros((len(learning_rate_list), n_eval))

    # Repeat experiments with diff learning rate
    np.random.seed(101210291)
    for i, learning_rate in enumerate(learning_rate_list):
        results[i] = repeatExperiments(learning_rate=learning_rate)

    plt.figure()
    x = np.arange(1, n_eval + 1) 
    for i, critic_step_size in enumerate(learning_rate_list):
        plt.plot(x, results[i], label=f"Learning Rate Size = {critic_step_size}")

    plt.xlabel("Episodes")
    plt.ylabel("Average Evaluated Return")
    plt.title("SARSA (with QNetwork Approx.) Performance")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{img_dest_path}/{file_name}.png")
    plt.clf()

    return results

def e_greedy(env, state, q_net, epsilon):
    """
        Returns e-greedy action for a given state
    """
    if torch.rand(1).item() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return greedy_policy(state, q_net)


def greedy_policy(state, q_net):
    """
        Returns best action for given state
    """
    q_values = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    action = torch.argmax(q_values).item()
    return action