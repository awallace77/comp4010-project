import numpy as np
from scipy.spatial.distance import cdist
import gymnasium as gym
import jax
'''
    utils.py
    Provides helper functions
'''
def evaluate_policy_fa(env, featurizer, W, policy_func, n_runs=10):
    '''
        Evaluate the policy given the parameters W and policy function.
        Run the environment several times and collect the return.
    '''
    all_returns = np.zeros([n_runs])
    for i in range(n_runs):
        observation, info = env.reset()
        return_to_go = 0
        while True:
            # Agent
            observation = featurizer.featurize(observation)
            action = policy_func(observation, W)

            observation, reward, terminated, truncated, info = env.step(action)
            return_to_go += reward
            if terminated or truncated:
                break
        all_returns[i] = return_to_go

    return np.mean(all_returns)

def evaluate_policy_nn(env, policy_func, n_runs=10):
    """
        Evaluate the policy given the policy function.
        Run the environment several times and collect the return.
    """
    
    all_returns = np.zeros([n_runs])
    for i in range(n_runs):
        observation, info = env.reset()
        return_to_go = 0
        while True:
            # Agent
            action = policy_func(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            return_to_go += reward
            if terminated or truncated:
                break
        all_returns[i] = return_to_go

    return np.mean(all_returns)


def render_env(env: gym.Env, featurizer, W, policy_func):
    """
        Renders a gym env with a provided featurizer
    """
    observation, info = env.reset()
    while True:
        env.render()
        observation = featurizer.featurize(observation)
        action = policy_func(observation, W)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Terminated or truncated")
            break

    env.close()
    return


def softplus(x, limit=30):
    """
       Numerically stable softplus function.
       Treat as linear when the input is larger than the limit.
    """
    return jax.numpy.where(x > limit, x, jax.numpy.log1p(jax.numpy.exp(x)))


def softplusGrad(x):
    """
       Gradient of the softplus function, which is sigmoid.
    """
    return jax.nn.sigmoid(x)


def logSoftmaxGradChecker(s, a, Theta, softmaxProb, analytic_grads):
    def grad_test_func(Theta):
        return jax.numpy.log(softmaxProb(s, Theta)[a])
    auto_grads = jax.grad(grad_test_func)(Theta)
    return np.allclose(analytic_grads, auto_grads)


def betaGradChecker(s, a, Theta, analytic_grads):
    def grad_test_func(s, Theta, a):
        alpha, beta = softplus(s @ Theta) + 1
        return jax.scipy.stats.beta.logpdf(a, alpha, beta)
    auto_grads = jax.grad(lambda Theta: grad_test_func(s, Theta, a))(Theta)

    print(analytic_grads[0])
    print(auto_grads[0])
    return np.allclose(analytic_grads, auto_grads)
