import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from rl.nnetworks.qnetwork import QNetwork


class ReplayMemory:
    def __init__(self, capacity: int, state_dim: int):
        #max number of transitions
        self.capacity = capacity 
        self.state_dim = state_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        #dones represents if the transition ended the episode
        self.dones = np.zeros((capacity,), dtype=np.float32)

        #pointer for the index of the information in the replay memory 
        self.ptr = 0 
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        index = self.ptr
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        #increase size untli capacity is reached
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indexes = np.random.choice(self.size, batch_size, replace=False)

        batch_states = torch.tensor(self.states[indexes], dtype=torch.float32)
        batch_actions = torch.tensor(self.actions[indexes], dtype=torch.int64)
        batch_rewards = torch.tensor(self.rewards[indexes], dtype=torch.float32)
        batch_next_states = torch.tensor(self.next_states[indexes], dtype=torch.float32)
        batch_dones = torch.tensor(self.dones[indexes], dtype=torch.float32)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        return self.size


def e_greedy(env, state, q_net, epsilon):
    if torch.rand(1).item() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return greedy_policy(state, q_net)

#purely greedy choosing the best action under the Q network
def greedy_policy(state, q_net):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
    return action


def dqn(
    env,
    eval_func,
    gamma=0.99,
    learning_rate=1e-3,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.005,
    max_episode=1000,
    evaluate_every=20,
    replay_capacity=50_000,
    batch_size=64,
    target_update_every=10,
):

    obs_shape = env.observation_space.shape
    if len(obs_shape) > 1:
        #flatten if observation is multi-dimensional
        state_dim = int(np.prod(obs_shape))
    else:
        state_dim = obs_shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(input_dim=state_dim, output_dim=n_actions)
    target_net = QNetwork(input_dim=state_dim, output_dim=n_actions)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    replay_memory = ReplayMemory(capacity=replay_capacity, state_dim=state_dim)

    #where to store evaluation returns
    n_eval = max_episode // evaluate_every
    eval_returns = np.zeros(n_eval, dtype=np.float32)
    eval_index = 0

    def preprocess_state(state):
        flat = np.array(state, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        return flat

    def get_epsilon(episode):
        #exponential decay in episodes
        return max(epsilon_min, epsilon_start * np.exp(-epsilon_decay * episode))

    for ep in range(max_episode):
        state, _ = env.reset()
        state = preprocess_state(state)

        done = False
        total_reward = 0.0
        epsilon = get_epsilon(ep)

        while not done:
            action = e_greedy(env, state, q_net, epsilon)

            #step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_state)

            #store transition
            replay_memory.push(state, action, reward, next_state, done)
            total_reward += reward

            state = next_state

            #learn from replay memory if there is enough samples
            if len(replay_memory) >= batch_size:
                (
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                ) = replay_memory.sample(batch_size)

                batch_states = batch_states
                batch_actions = batch_actions
                batch_rewards = batch_rewards
                batch_next_states = batch_next_states
                batch_dones = batch_dones

                #current Q-values: Q(s,a)
                q_values = q_net(batch_states)
                q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                #target Q-values: r + γ * max_a' Q_target(s',a') * (1 - done)
                with torch.no_grad():
                    next_q_values = target_net(batch_next_states)
                    max_next_q_values, _ = torch.max(next_q_values, dim=1)
                    targets = batch_rewards + gamma * max_next_q_values * (1.0 - batch_dones)

                #loss and optimization step
                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #periodically update target network
        if (ep + 1) % target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        #periodically evaluate the greedy policy with the current network
        if (ep + 1) % evaluate_every == 0:
            def policy_fn(obs):
                s_proc = preprocess_state(obs)
                return greedy_policy(s_proc, q_net)

            avg_return = eval_func(env, policy_func=policy_fn, n_runs=10)
            eval_returns[eval_index] = avg_return
            eval_index += 1

    return q_net, eval_returns


def run_dqn_experiments(env, eval_func, img_dest_path: str = "", file_name: str = ""):

    def repeat_experiments(learning_rate):
        eval_returns_lr = np.zeros((n_runs, n_eval), dtype=np.float32)
        for r in range(n_runs):
            q_net, eval_returns = dqn(
                env,
                eval_func=eval_func,
                gamma=0.99,
                learning_rate=learning_rate,
                max_episode=max_episodes,
                evaluate_every=evaluate_every,
            )
            eval_returns_lr[r] = eval_returns

        return np.mean(eval_returns_lr, axis=0)

    n_runs = 10
    max_episodes = 2000
    evaluate_every = 20
    n_eval = max_episodes // evaluate_every

    learning_rate_list = [0.001, 0.0005, 0.00025]
    results = np.zeros((len(learning_rate_list), n_eval), dtype=np.float32)

    np.random.seed(101232374)
    for i, lr in enumerate(learning_rate_list):
        results[i] = repeat_experiments(learning_rate=lr)

    plt.figure()
    x = np.arange(1, n_eval + 1)
    for i, lr in enumerate(learning_rate_list):
        plt.plot(x, results[i], label=f"Learning rate = {lr}")

    plt.xlabel("Evaluation index (every 20 episodes)")
    plt.ylabel("Average evaluated return")
    plt.title("DQN Performance")
    plt.grid(True)
    plt.legend()

    if img_dest_path and file_name:
        plt.savefig(f"{img_dest_path}/{file_name}.png")
        plt.clf()
    else:
        plt.show()

    return results