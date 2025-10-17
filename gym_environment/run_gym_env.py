import gymnasium as gym
import gymnasium_env

env = gym.make('gymnasium_env/GridWorld-v0', render_mode = 'human')

observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:

    action = env.action_space.sample()  

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()