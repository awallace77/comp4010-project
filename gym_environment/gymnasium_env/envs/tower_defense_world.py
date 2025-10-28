import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
from dataclasses import dataclass
from gymnasium_env.envs.tower_defense_logic import TowerDefenseGame
import math
"""
    COMP4010: Intro to RL
    Project Environment Demo
    Group Group 12
        Andrew Wallace - 101210291 - andrewwallace3@cmail.carleton.ca
        Mohammad Rehman - 101220514 - mohammadrehman@cmail.carleton.ca
        Manal Hassan - 101263813 - manalhassa@cmail.carleton.ca
        Derrick Zhang - 101232374 - derrickzhang@cmail.carleton.ca

    Due: October 27th, 2025
"""

'''
    TODO:
        - Handle base destroyed
        - Add additional towers
        - Add multiple enemies
        - Add tower level up mechanism
        - Update UI to have HP 
        - Additional algos
        - Add money logic
        etc., 
'''


@dataclass(frozen=True)
class Rewards:
    ENEMY_DEFEATED: int = 10

    TOWER_DEFEATED: int = -5
    TOWER_DAMAGED: int = -1 

    ENEMY_REACH_BASE: int = -50
    TOWER_LEVEL_UP: int = 5
    WAVE_CLEARED: int = 20
    ALL_WAVES_CLEARED: int = 200
    BASE_DESTROYED: int = -10

@dataclass(frozen=True)
class GridCell:
    EMPTY: int = 0
    TOWER: int = 1
    ENEMY: int = 2
    PATH: int = 3

class TowerDefenseWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, size=5, num_enemies=3):

        # General setup
        self.size = size
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Wave setup
        self.wave_count = 0
        self.max_waves = 3

        # Rewards
        self.rewards = Rewards()

        # Tower setup
        self.tower_hp = 28
        self.tower_damage = math.floor(math.log2(self.tower_hp))
        self.tower_types = {"st": 0, "aoe": 1}

        # Enemy setup
        self.num_enemies = num_enemies
        self.enemy_hp = 13
        self.enemy_damage= math.floor(math.log2(self.enemy_hp))

        # Gameplay setup
        self.phase = "build" # Phases: 'build' or 'defend'
        self.grid_towers = np.zeros((size, size), dtype=int)  # 0 = empty, 1 = tower
        self.grid_enemies = np.zeros((size, size), dtype=int) # 0 = empty, 2 = enemy, 3=path 

        # using the existing s shaped path code
        self.path = TowerDefenseGame.s_path(size)
        for p in self.path:
            self.grid_enemies[p] = GridCell.PATH

        # Towers and enemies info
        self.towers = {}  # Dict with (y, x): {hp: number, damage: number, type: string}
        self.enemies = []  # List of dicts with enemy {pos, hp, damage}
        self.base = self.path[-1]  # Last cell in path

        self.action_space = spaces.Discrete(size * size + 1)  # 0 = do nothing, others = place at specific cell

        # Enemy actions: 0=do nothing, 1=down, 2=left, 3=right
        self.enemy_action_space = spaces.Discrete(4) 

        # Observation (state space): a 3d array (matrix) where
        '''[i, j, k]: 
            # i = the row (y)
            # j = the column (x)
            # k = information for path, tower hp, and enemy hp
            i.e., 
                k[0] = 1 if path
                k[1] = tower hp if tower exists at [i,j]
                k[2] = enemy hp if enemy exists at [i,j]
        '''
        self.observation_space = spaces.Box(low=0, high=max(self.enemy_hp, self.tower_hp), shape=(self.size, self.size, 3), dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # NOTE: Eventually will update to have towers persist through episodes
        
        # Clear all info and set to original state
        self.grid_enemies[:] = GridCell.EMPTY
        self.grid_towers[:] = GridCell.EMPTY
        for p in self.path:
            self.grid_enemies[p] = GridCell.PATH
        self.towers.clear() 
        self.enemies.clear()
        self.phase = "build"
        self.wave_count = 0

        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        e_defeated = 0
        t_defeated = 0

        # Beginning of wave
        if self.phase == "build":
            reward = self._build_phase_step(action)
            self.phase = "defend"
            self._spawn_enemies()

        # During a wave
        elif self.phase == "defend":
            reward, terminated, e_defeated, t_defeated = self._defense_phase_step()
            if terminated:
                self.wave_count += 1

                # Completed all waves
                if self.wave_count >= self.max_waves:
                    reward += self.rewards.ALL_WAVES_CLEARED
                    terminated = True
                
                else:
                    # Start next wave
                    self.phase = "build"
                    terminated = False
                    reward += self.rewards.WAVE_CLEARED

        obs = self._get_obs()
        info = {"phase": self.phase, "wave": self.wave_count, "enemies_destroyed": e_defeated, "towers_destroyed": t_defeated}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, info

    def _build_phase_step(self, action):
        """
            Handles beginning of wave actions
            Args:
                The initial action chosen
            Returns:
                A potential build phase reward
        """
        if action == 0:
            # NOTE: Small neg reward for doing nothing?
            return 0  

        # Map action number to grid position
            # divmod(action, self.size) converts the flat action index into a 2D grid coordinate (y, x).
        action -= 1  # because 0 is "do nothing"
        y, x = divmod(action, self.size)

        # Place (create) tower if valid
        if self.grid_towers[y, x] != GridCell.TOWER: 
            self.grid_towers[y, x] = GridCell.TOWER

            # Set tower health, damage, and type
            self.towers[(y, x)] = {"hp": self.tower_hp, "damage": self.tower_damage, "type": self.tower_types["st"]}

        # NOTE: Potential rewards in initial build phase? Something with the budget maybe?
        return 0


    def _spawn_enemies(self):
        """
            Spawn enemies at the start of the path.
            NOTE: To update to handle multiple paths
        """
        for i in range(self.num_enemies):
            self.enemies.append({"pos": list(self.path[0]), "hp": self.enemy_hp, "damage": self.enemy_damage})
            self.grid_enemies[tuple(self.path[0])] = GridCell.ENEMY

    def _defense_phase_step(self):
        """
            Simulates a single tower defense step
            Returns:
                A tuple containing the reward value, boolean terminated, the number of enemies destroyed, and number of towers destroyed
        """
        reward = 0
        terminated = False
        e_defeated = 0
        t_defeated = 0

        for enemy in self.enemies:
            ey, ex = enemy["pos"]
            e_damage = enemy["damage"]
            # Get enemy next cell on path
            next_idx = self.path.index((ey, ex)) + 1
            if next_idx < len(self.path):
                next_pos = self.path[next_idx]
                ny, nx = next_pos
                if self.grid_towers[ny, nx] > 0:
                    # Tower blocks the enemy: deal damage to tower instead of moving
                    self.towers[(ny, nx)]["hp"] -= e_damage

                    # Negative reward for taking damage to encourage better placement
                    reward += self.rewards.TOWER_DAMAGED
                else:
                    # Move forward
                    enemy["pos"] = [ny, nx]
            else:
                reward += self.rewards.ENEMY_REACH_BASE # negative reward for enemy reaching base
                terminated = True

        # Towers attack along their row and column
        for (ty, tx), tower in self.towers.items():
            t_damage = tower["damage"]
            for enemy in self.enemies:
                ey, ex = enemy["pos"]
                if ey == ty or ex == tx:  # same row or same column
                    enemy["hp"] -= t_damage

        # Remove dead enemies
        e_count = len(self.enemies)
        self.enemies = [e for e in self.enemies if e["hp"] > 0]
        e_defeated = e_count - len(self.enemies)
        reward += e_defeated * self.rewards.ENEMY_DEFEATED

        # Remove dead towers
        t_count = len(self.towers.keys())
        dead_positions = [pos for pos, t in self.towers.items() if t["hp"] <= 0]
        for pos in dead_positions:
            del self.towers[pos]
        t_defeated = t_count - len(self.towers.keys())
        reward += t_defeated * self.rewards.TOWER_DEFEATED

        # Update tower grid
        self.grid_towers[:] = GridCell.EMPTY
        for (ty, tx), tower in self.towers.items():
            self.grid_towers[ty, tx] = GridCell.TOWER

        # Update enemy grid
        self.grid_enemies[:] = GridCell.EMPTY
        for p in self.path:
            self.grid_enemies[p] = GridCell.PATH
        for enemy in self.enemies:
            y, x = enemy["pos"]
            self.grid_enemies[y, x] = GridCell.ENEMY

        # Check if all enemies are destroyed
        if not self.enemies:
            terminated = True
            reward += self.rewards.WAVE_CLEARED

        return reward, terminated, e_defeated, t_defeated

    def _get_obs(self):
        obs = np.zeros((self.size, self.size, 3), dtype=int)
    
        # Layer 0: path
        for y, x in self.path:
            obs[y, x, 0] = 1 # 1 if cell is a part of a path
        
        # Layer 1: towers (store HP)
        for (y, x), tower in self.towers.items():
            obs[y, x, 1] = tower["hp"]
        
        # Layer 2: enemies (store HP)
        for enemy in self.enemies:
            y, x = enemy["pos"]
            obs[y, x, 2] = enemy["hp"]
        
        obs = self.state_to_key(obs)
        return obs

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.render_mode != "human":
            return

        cell_size = 80
        window_size = self.size * cell_size
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((window_size, window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))

        color_map = {
            GridCell.EMPTY: (240, 240, 240),  # empty
            GridCell.TOWER: (0, 150, 0),      # tower
            GridCell.ENEMY: (200, 0, 0),      # enemy
            GridCell.PATH: (150, 150, 150),  # path
        }

        # Draw the grid
        for y in range(self.size):
            for x in range(self.size):
                # Draw path or empty background
                if (y, x) in self.path:
                    pygame.draw.rect(canvas, color_map[GridCell.PATH], pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size))
                else:
                    pygame.draw.rect(canvas, color_map[GridCell.EMPTY], pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size))

                # Draw enemies first (so tower can be on top)
                for enemy in self.enemies:
                    ey, ex = enemy["pos"]
                    if ey == y and ex == x:
                        pygame.draw.rect(canvas, color_map[GridCell.ENEMY], pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size))

                # Draw tower on top
                for (ty, tx) in self.towers.keys():
                    if ty == y and tx == x:
                        pygame.draw.circle(
                            canvas,
                            color_map[GridCell.TOWER],
                            (x*cell_size + cell_size//2, y*cell_size + cell_size//2),
                            cell_size//3
                        )

        # Draw grid lines
        for i in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (i*cell_size, 0), (i*cell_size, window_size))
            pygame.draw.line(canvas, (0, 0, 0), (0, i*cell_size), (window_size, i*cell_size))

        # Blit canvas
        self.window.blit(canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
 

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.window = None
            
    def state_to_key(self, obs):
        # obs is a numpy array shape (size, size, 3). Convert to an immutable tuple key.
        return tuple(obs.flatten().tolist())

if __name__ == "__main__":

    # Local testing
    env = TowerDefenseWorld(render_mode="human")
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # random policy
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # draw the grid
        env.render()          
        time.sleep(0.3)

    print("Episode finished with total reward:", total_reward)
 