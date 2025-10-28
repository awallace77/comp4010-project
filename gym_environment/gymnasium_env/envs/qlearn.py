import numpy as np
import random
import pygame
import time
from collections import defaultdict
from tower_defense_world import TowerDefenseWorld
"""
    COMP4010: Intro to RL
    Project Environment Demo
    Group Group 12
        Andrew Wallace - 101210291 - andrewwallace3@cmail.carleton.ca
        Mohammad Rehman - 101220514 - mohammadrehman@cmail.carleton.ca
        Manal Hassan - 101263813 - manalhassa@cmail.carleton.ca
        Derrick Zhang - 101232374 - derrickzhang@cmail.carleton.ca
    Due: October 27th, 2025

    Below is the implementation of the q-learning algorithm for our Tower Defense Environment.
"""

def q_learning(env,
                     episodes=2000,
                     alpha=0.5,
                     gamma=0.95,
                     epsilon_start=0.9,
                     epsilon_end=0.05,
                     epsilon_decay_steps=1500,
                     log=False):
    """
    Q-learning on the provided environment
    Args:
        env: env to learn on 
        episodes: number of training episodes
        alpha: learning rate
        gamma: discount factor
        epsilon: epsilon-greedy parameter 
        log: log results
    Returns:
        Approximated Q values
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def e_greedy(e, state):
        if random.random() < e:
            return env.action_space.sample()
        else:
            return int(np.argmax(Q[state]))

    epsilon = epsilon_start
    eps_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)

    for ep in range(1, episodes + 1):
        state, info = env.reset()
        terminated = False
        total_reward = 0
        
        while not terminated:
            action = e_greedy(epsilon, state)
            state_prime, reward, terminated, truncated, info = env.step(action)

            # Q-learning update
            best_next = np.max(Q[state_prime])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state = state_prime
            total_reward += reward

        # Decay epsilon
        if ep <= epsilon_decay_steps:
            epsilon = max(epsilon_end, epsilon - eps_decay)
        else:
            epsilon = epsilon_end

        # Log results of episode (every 100th)
        if log and ep % 100 == 0 or ep == 1:
            print(f"Episode {ep} of {episodes} total_reward={total_reward:.2f} eps={epsilon:.3f}")
            print(f"Wave {info['wave']} enemies_destroyed={info['enemies_destroyed']}  towers_destroyed={info['towers_destroyed']}\n")

    return Q

def evaluate_policy(env, Q, episodes=10, sleep=0.1):

    print("EVALUATION OF Q-LEARNED POLICY **** \n")

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:

            action = int(np.argmax(Q[state])) # greedy action derived from Q
            state, reward, terminated, _, info = env.step(action)
            done = terminated
            total_reward += reward

            env.render()
            pygame.time.wait(300)

        print(f"Episode {ep+1}/{episodes}  total_reward={total_reward:.2f}")
        print(f"Wave {info.get('wave', '?')} enemies_destroyed={info.get('enemies_destroyed', '?')}  towers_destroyed={info.get('towers_destroyed', '?')}\n")

    env.close()

if __name__ == "__main__":

    # env = TowerDefenseWorld(render_mode="human") 
    env = TowerDefenseWorld(render_mode=None)
    
    # Approximate Q values
    Q = q_learning(env, 
                   episodes=10000, 
                   alpha=0.5, 
                   gamma=0.9, 
                   epsilon_start=0.9, 
                   epsilon_end=0.05,
                   epsilon_decay_steps=1500,
                   log=True)
    env.close()

    env = TowerDefenseWorld(render_mode="human") 
    evaluate_policy(env, Q, episodes=10, sleep=0.5)
    env.close()
