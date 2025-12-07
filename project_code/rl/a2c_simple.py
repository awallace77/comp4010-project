import numpy as np
from matplotlib import pyplot as plt
from rl.utils import *
import math
"""
    Advantage Actor Critic (A2C) w TD(0)
"""
def a2c_simple(
        env,
        featurizer,
        eval_func,
        gamma=0.99,
        actor_step_size=0.00005,
        critic_step_size=0.0001,
        max_episodes=1000,
        evaluate_every=50):
    """
        Advantage Actor-Critic (A2C) using TD(0)
        Args:
            env: a gym environment
            featurizer: converts a state S of the env into a feature vector
            eval_func: an evaluation function 
            gamma: scalar discount factor in [0, 1]
            actor_step_size: scalar step size for actor gradient updates
            critic_step_size: scalar step size for critic gradient updates
            max_episodes: integer indicating th max number of episodes before the algo terminates
            evaluate_every: integer indicating how frequent to evaluate
        Returns:
            The learned actor parameters Theta, critic parameters w, and a Python list of runs evaluated during training
    """ 
    # Extract sizes
    a_space_size = env.action_space.n 
    d = featurizer.n_features

    # Initialize parameters
    # Theta = np.random.rand(d, a_space_size)
    # w = np.random.rand(d)
    Theta = np.random.randn(d, a_space_size) * 0.01
    w = np.zeros(d)

    eval_returns = []
    for i in range(1, max_episodes + 1):
        s, _ = env.reset()
        s = featurizer.featurize(s)
        terminated = truncated = False
        # actor_discount = 1
        while not (terminated or truncated):
            
            # Value of current state 
            s_value = s.T @ w

            # Choose and take action according to softmax policy
            a = softmaxPolicy(s, Theta)
            s_prime, reward, terminated, truncated, _ = env.step(a)

            # Featurize new state
            s_prime = featurizer.featurize(s_prime)

            # State value & gradient of next state
            s_prime_value = 0 if (terminated or truncated) else s_prime.T @ w
            
            # TD error w critic estimate
            target = reward + gamma * s_prime_value
            td_error = target - s_value

            # Semi-gradient update critic
            w = w + critic_step_size * td_error * s 

            # Policy gradient update actor 
            policy_gradient = logSoftmaxPolicyGradient(s, a, Theta)
            # Theta = Theta + actor_step_size * td_error * actor_discount * policy_gradient
            Theta = Theta + actor_step_size * td_error * policy_gradient

            # actor_discount *= gamma

            s = s_prime

        if i % evaluate_every == 0:
            eval_return = eval_func(env, featurizer, Theta, softmaxPolicy)
            eval_returns.append(eval_return)
            # print(f"[ActorCritic evaluation]: {i}th episode")

        if i % math.floor(max_episodes / 5) == 0:
            print(f"[A2C]: Episode {i}; Theta {Theta}; w {w}")

    return Theta, w, eval_returns


def run_a2c_simple_experiments(env, featurizer, eval_func, img_dest_path="", file_name=""):    
    """
        Runs experiments for ActorCritic
        Args:
            env: A gym environment
            featurizer: converts the state/observation S of the env into a feature vector of dimension d
            eval_func: an evaluation function that evaluates the performance of the intermediate policy during training
            img_dest_path: absolute path to save image of results to 
            file_name: the name of the resulting image
        Returns:
            results: the results of the experiments
    """

    def repeatExperiments(actor_step_size, critic_step_size):
        eval_returns_step_sizes = np.zeros([n_runs, n_eval])
        for r in range(n_runs):
            Theta, w, eval_returns = a2c_simple(
                env,
                featurizer,
                eval_func,
                actor_step_size=actor_step_size,
                critic_step_size=critic_step_size,
                max_episodes=max_episodes,
                evaluate_every=evaluate_every)

            eval_returns_step_sizes[r] = eval_returns
        
        # Compute and return the average evaluated returns over runs
        avg_eval_returns = np.mean(eval_returns_step_sizes, axis=0) 
        return avg_eval_returns

    n_runs = 1
    max_episodes = 10000
    evaluate_every = 100
    n_eval = max_episodes // evaluate_every # num of evaluation during training

    actor_step_size = 0.00005
    # actor_step_size = 0.0001
    critic_step_size_list = [0.005, 0.001, 0.0005, 0.0001]
    results = np.zeros([len(critic_step_size_list), n_eval])

    np.random.seed(101210291)

    # Repeat experiments with different critic_step_size
    for i, critic_step_size in enumerate(critic_step_size_list):
        results[i] = repeatExperiments(actor_step_size=actor_step_size, critic_step_size=critic_step_size)

    plt.figure()
    x = np.arange(1, n_eval + 1) 
    for i, critic_step_size in enumerate(critic_step_size_list):
        plt.plot(x, results[i], label=f"Critic step size = {critic_step_size}")

    plt.xlabel(f"Evaluation Number (Every {evaluate_every} Episodes)")
    plt.ylabel("Average Evaluated Return")
    plt.title(f"A2C with Linear Approximation (Actor Step Size = {actor_step_size})\nAverage Return over {n_runs} runs ({max_episodes} episodes per run)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{img_dest_path}/{file_name}_{n_runs}runs_{max_episodes}episodes.png")
    plt.clf()

    return results

def softmaxProb(x, Theta):
    """
        Calculates the softmax probabilities over the actions
        Args:
            x: d-by-1 state feature vector
            Theta: d-by-|A| parameters of the actor
        Returns: 
            The softmax probabilities over the actions
    """

    # State action preference h(s)
    h = Theta.T @ x

    # Max state action preference
    m = np.max(h)
   
    # Construct denominator
    probs = None
    total_h = np.sum(np.exp(h - m))
    probs = np.exp((h - m)) / total_h

    return probs 

def softmaxPolicy(x, Theta):
    """
        Args:
            x: d-by-1 state feature vector
            Theta: d-by-|A| matrix of the actor
        Returns:
            An action sampled from the softmax probabilities
    """
    probs = softmaxProb(x, Theta)

    # Sample action based on softmax probs
    a = np.random.choice(len(probs), p=probs)
    return a

def logSoftmaxPolicyGradient(x, a, Theta):
    """
        Calculates the gradient of a chosen action w.r.t the parameters
        Args:
            x: d-by-1 state feature vector
            a: integer action
            Theta: d-by-|A| matrix of the actor's parameters
        Returns:
            The gradient of chosen action w.r.t the parameters Theta (d-by-|A|)
    """

    # Get action probabilities
    probs = softmaxProb(x, Theta)
    _, a_size = Theta.shape

    # One-hot for chosen action
    one_hot = np.zeros(a_size)
    one_hot[a] = 1.0

    # Columns without feature x for respective column gradient
    diff = one_hot - probs

    # Multiply each entry in feature vector by each difference in diff to construct dx|A|
    gradient = np.outer(x, diff)

    assert gradient.shape == Theta.shape
    return gradient