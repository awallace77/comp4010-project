import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game.game_info import TowerInfo, EnemyInfo, BaseInfo, BudgetInfo
from enum import IntEnum, StrEnum
from dataclasses import dataclass
from game.entities.enemy import Enemy
from game.entities.tower import SingleTargetTower, AoETower, Tower
from game.entities.base import Base
import math
"""
    COMP4010: Intro to RL
    Project Environment Demo
    Group Group 12
        Andrew Wallace - 101210291 - andrewwallace3@cmail.carleton.ca
        Mohammad Rehman - 101220514 - mohammadrehman@cmail.carleton.ca
        Manal Hassan - 101263813 - manalhassa@cmail.carleton.ca
        Derrick Zhang - 101232374 - derrickzhang@cmail.carleton.ca
"""

'''
    General TODOs:
        - DONE: Handle base destroyed
        - DONE: Update UI to have HP
        - TODO: Add Additional RL algos (from libraries)
        etc., 

        Andrew TODOs:
            - DONE: Add additional towers
            - DONE: Add multiple enemies
            - DONE: Add tower level up mechanism -- DONE
            - DONE: Update UI to have tower levels and damage
            - TODO: Update state to handle multiple towers and enemies
            - TODO: Add money/budget logic
            - TODO: Allow for placing multiple towers and moving existing towers
'''

ENV_NAME = "RL Tower Defense | COMP4010 - Group 12"
MAX_INT = 2**63 - 2 # max possible int for spaces.Box

@dataclass(frozen=True)
class Reward:
    """Rewards for the environment"""
    ENEMY_DEFEATED: int = 10
    ENEMY_REACH_BASE: int = -50

    TOWER_DEFEATED: int = -5
    TOWER_DAMAGED: int = -1 
    TOWER_LEVEL_UP: int = 5

    WAVE_CLEARED: int = 20
    ALL_WAVES_CLEARED: int = 200
    BASE_DESTROYED: int = -10

class GridCell(IntEnum):
    """The different grid cell types"""
    TOWER = 0
    ENEMY = 1
    BASE = 2
    PATH = 3
    EMPTY = 4

class Phase(StrEnum):
    """Used to specify the current phase"""
    BUILD = "build"
    DEFEND = "defend"


class TowerDefenseEnv(gym.Env):
    """
        The tower defense environment is an nxn grid consisting of towers, enemies, a base, and paths for the enemies to take to the base.
        Towers and enemies attack each other. Enemy goal is to reach the base and destroy it.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
            self,  
            size=5, 
            num_enemies=1,
            max_waves=10, 
            render_mode=None,
            render_rate=100
        ):

        # General setup
        self.size = size
        self.render_mode = render_mode
        self.render_rate = render_rate
        self.window = None
        self.clock = None

        # Wave setup
        self.wave_count = 1
        self.max_waves = max_waves

        # Gameplay setup
        self.phase = Phase.BUILD
        self.total_reward = 0
        self.current_episode = 0

        # Budget setup
        self.start_budget = BudgetInfo.BUDGET
        self.budget = BudgetInfo.BUDGET
        
        # Grid setup
        #[i, j, k] - i, j is (row, col), 
            # k[0] = 1 if tower, 0 otherwise 
            # k[1] = 1 if enemy, 0 otherwise
            # k[2] = 1 if base, 0 otherwise
            # k[3] = 1 if path, 0 otherwise
        self._reset_grid()
        
        # Tower setup
        self.towers: dict[tuple, Tower] = {}  # Dict with (y, x): Tower object

        # Enemy setup
        self.num_enemies = num_enemies
        self.enemies: list[Enemy] = []  # List of enemy objects 
        self.path = self.s_path(size) # Path each enemy to follow

        # Base setup
        self.base =  Base(pos=(self.size-1, self.size-1), health=BaseInfo.HEALTH) 
        by, bx = self.base.pos
        self.grid[by, bx, GridCell.BASE] = GridCell.BASE

        # Action space setup
        # 0 = do nothing, others = place tower at specific cell
        # TODO: Update to enable moving towers/placing multiple towers in accordance to budget at beginning of wave
        self.action_space = spaces.Discrete(2*(size * size) + 1) 

        # Observation (state space) setup 
        '''A 3d array (n x n x 8) where [i, j, k]: 
            # i = the row (y)
            # j = the column (x)
            # k = information for cell (see _get_obs for more details)
        '''
        self.observation_space = spaces.Dict({
            "grid": spaces.Box( # Grid info
                low=0,
                high=max(TowerInfo.MAX_LEVEL, EnemyInfo.MAX_HEALTH, BaseInfo.MAX_HEALTH),
                shape=(self.size, self.size, 8),
                dtype=int
            ),
            "budget": spaces.Box( # Current Budget
                low=0,
                high=BudgetInfo.MAX_BUDGET,
                shape=(1,),
                dtype=int
            )
        })


    def reset(self, seed=None, options=None):
        """
            Resets the environment
            Args:
                seed: The seed that is used to initialize the environment's PRNG
                options: Additional information to specify how the environment is reset
        """
        super().reset(seed=seed)
        
        # Clear all info and set to original state
        self._reset_grid()
        self.towers.clear() 
        self.enemies.clear()
        self.base.respawn()
        self.phase = Phase.BUILD
        self.wave_count = 0
        self.current_episode += 1
        self.budget = self.start_budget
        self._update_grid()

        return self._get_obs(), {}


    def step(self, action):
        """
            Takes step from given action. Step involves both build and defend phases.
            Args:
                action: the action to take
            Returns:
                A tuple of:
                  obs: the next observation (state), 
                  cumulative_reward: reward collected during this step
                  terminated: whether the episode reaches a terminal state
                  truncated: whether the episode ended due to external constraint
                  info: additional information about the step
        """
        cumulative_reward = 0 # reward collected from each step within a wave
        terminated = False
        truncated = False
        e_defeated = 0

        # Beginning of wave
        if self.phase == Phase.BUILD:
            cumulative_reward += self._build_phase_step(action)
            self._spawn_enemies()
            self.phase = Phase.DEFEND 

        # During a wave (one wave represents a single step)
        if self.phase == Phase.DEFEND:

            wave_terminated = False

            while (not wave_terminated) and self.wave_count < self.max_waves:

                reward, wave_terminated, e_defeated = self._defense_phase_step()
                cumulative_reward += reward

                if self.render_mode == "human": # Render wave
                    self.render()
                    pygame.time.wait(self.render_rate)

                # Wave terminated
                if wave_terminated:
                    self.wave_count += 1
                    self.phase = Phase.BUILD 

        # Terminated conditions (terminates an episode)
        terminated = False
        if self.wave_count >= self.max_waves or self.base.is_dead():
            terminated = True
            cumulative_reward += ( # Final Rewards
                (self.wave_count >= self.max_waves) * Reward.ALL_WAVES_CLEARED
            )

        obs = self._get_obs()
        info = {
            "phase": self.phase, 
            "wave": self.wave_count, 
            "enemies_destroyed": e_defeated, 
            "base_destroyed": self.base.is_dead(),
            "base_health": self.base.health,
            "base_start_health": self.base.original_health
        }

        self.total_reward += cumulative_reward
        return obs, cumulative_reward, terminated, truncated, info


    def _build_phase_step(self, action):
        """
            Handles beginning of wave actions
            Args:
                action: the initial action chosen in [0, 2*size*size]
            Returns:
                A build phase reward
        """
        # Action can now be [0, 2*size*size]
        original_action = action
        reward = 0

        if action == 0:
            # TODO: Small neg reward for doing nothing?
            return reward

        # Map action number to grid position 
        action -= 1  # get rid of 0 because 0 is "do nothing"

        if action >= self.size * self.size:
            action = math.floor(action / 2)

        y, x = self._action_to_coordinate(self.size, action)

        # TODO: add option to place multiple towers based on current budget
        # Need to encode an action to include number of towers and type of tower
        # ()

        # Place (create) tower if valid - must not be on path or already have a tower
        if (y, x) != self.base.pos and (y, x) not in self.path and self.grid[y, x, GridCell.TOWER] != GridCell.TOWER: 

            if original_action >= self.size * self.size:
                tower = SingleTargetTower(pos=(y,x), health=TowerInfo.SINGLE_TARGET_HEALTH)
            else:
                tower = AoETower(pos=(y,x), health=TowerInfo.AOE_HEALTH)

            self.towers[tower.pos] = tower 
            self.grid[y, x, GridCell.TOWER] = GridCell.TOWER
            
        return reward


    def _spawn_enemies(self):
        """
            Spawn enemies at the start of their given path.
            TODO: To update to handle multiple (different) paths
        """
        for _ in range(self.num_enemies):
            # Generate path and create enemy
            # path = self.s_path(self.size)

            ## This creates random paths for an enemy 
            # path = self._generate_enemy_path(self.size, self.base.pos)

            path = self.path # set default path
            enemy = Enemy(pos=path[0], path=path)
            self.enemies.append(enemy)
            y, x = enemy.pos
            self.grid[y, x, GridCell.ENEMY] = GridCell.ENEMY


    def _defense_phase_step(self):
        """
            Simulates a single tower defense step
            Returns:
                A tuple containing the reward value, boolean terminated, the number of enemies destroyed, and number of towers destroyed
        """
        reward = 0
        wave_terminated = False
        e_defeated = 0
        e_damaged = 0
        enemies_to_remove = []

        # Move enemies along the path (they don't interact with towers while moving)
        for i, enemy in enumerate(self.enemies):
            
            reached_base = enemy.move()
            if reached_base:
                self.base.take_damage(enemy.damage)
                reward += Reward.ENEMY_REACH_BASE
                enemies_to_remove.append(i)

        # Remove enemies that reached the base
        for idx in reversed(enemies_to_remove):
            self.enemies.pop(idx)

        # Check if base is destroyed
        if self.base.is_dead():
            reward += Reward.BASE_DESTROYED  # additional penalty
            wave_terminated = True
            return reward, wave_terminated, e_defeated 

        # Towers attack 
        for tower in self.towers.values():
            attacked_enemies = tower.attack(self.enemies)
            e_damaged += len(attacked_enemies)

            for e in attacked_enemies: # level up towers
                if e.is_dead():
                    leveled_up = tower.killed_enemy()
                    if leveled_up: 
                        reward += Reward.TOWER_LEVEL_UP

        # Remove dead enemies
        e_count = len(self.enemies)
        self.enemies = [e for e in self.enemies if not e.is_dead()]
        e_defeated = e_count - len(self.enemies)

        self._reset_grid()
        self._update_grid() 
        
        if not self.enemies: # check if wave terminated (no more enemies)
            wave_terminated = True
            reward += Reward.WAVE_CLEARED

        reward += ( # add up all rewards during defense phase
            e_defeated * Reward.ENEMY_DEFEATED
        )

        return reward, wave_terminated, e_defeated 


    def _get_obs(self, as_tuple=True):
        """Returns the current observation (state)"""
        obs = np.zeros((self.size, self.size, 8), dtype=int)
        
        # Tower info
        for (y, x), tower in self.towers.items():
            obs[y, x, 0] = tower.get_id()
            obs[y, x, 1] = tower.level
        
        # Enemy info
        for enemy in self.enemies:
            y, x = enemy.pos
            obs[y, x, 2] += 1 # num_enemies 
            # obs[y, x, 3].append(enemy.get_id()) # types of enemies TODO: If adding different enemies, update state
            obs[y, x, 3] = (obs[y, x, 3] + enemy.health) / obs[y, x, 2] # avg enemy health
            obs[y, x, 4] = max(obs[y, x, 4], enemy.health) # max hp of enemies

            # Layer 3: Path
            for py, px in enemy.path:
                obs[py, px, 5] = 1 # enemy path

        # Layer 2: Base
        if self.base is not None:
            y, x = self.base.pos
            obs[y, x, 6] = 1  # base or not
            obs[y, x, 7] = self.base.health  # base health 

        if as_tuple:
            return self.state_to_key(obs)

        return obs


    def render(self):
        """ Renders current state to pygame display """
        return self._render_frame() 
 

    def close(self):
        """ Closes the environment """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.window = None

            
    def state_to_key(self, obs):
        """ Converts observation to key """
        return tuple(obs.flatten().tolist())    


    def s_path(self, n: int, gap: int = 2):
        """    
            Lays down coordinates for an S shape based on grid size
            Args:
                n: the size of the nxn grid
                gap: the number of rows to skip (including current row)
            Returns:
                An array containing an s shaped path of coordinates (row, col)
        """
        path = []
        r = 0
        left_to_right = True
        while r < n:
            if left_to_right:
                cols = range(0, n)
            else:
                cols = range(n - 1, -1, -1)
            for c in cols:
                path.append((r, c))
            r += gap
            left_to_right = not left_to_right
        return path
    
    def add_tower_to_grid(self, tower):
        """Add a tower object to the grid"""
        self.towers[tower.pos] = tower
        self._update_grid()

    
    def _generate_enemy_path(self, n: int, goal: tuple[int, int]):
        """Generate a path from goal to any edge of the grid."""
        y, x = goal
        path = [(y, x)]
        seen = {(y, x)}

        # Define possible movement directions (N, S, E, W)
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # up, down, right, left

        def is_edge(y, x):
            return y == 0 or y == n - 1 or x == 0 or x == n - 1

        def dfs(y, x):
            if is_edge(y, x):
                return True
            
            np.random.shuffle(directions)  # randomize path structure
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < n and 0 <= nx < n and (ny, nx) not in seen:
                    seen.add((ny, nx))
                    path.append((ny, nx))
                    if dfs(ny, nx):  # continue until an edge is found
                        return True
                    path.pop()  # backtrack if dead end
            
            return False  # no valid move found

        dfs(y, x)
        path.reverse()
        return path 
    

    def _reset_grid(self):
        """ Resets grid to "original" empty state """
        self.grid = -np.ones((self.size, self.size, len(GridCell) - 1), dtype=int)


    def _update_grid(self):
        """ Update grid with current info """
        for (ty, tx), tower in self.towers.items(): # Update towers on grid
            self.grid[ty, tx, GridCell.TOWER] = GridCell.TOWER # type of tower
        
        for enemy in self.enemies: # Update enemies on grid
            y, x = enemy.pos
            self.grid[y, x, GridCell.ENEMY] = GridCell.ENEMY

            for (py, px) in enemy.path: # Update paths on grid
                self.grid[py, px, GridCell.PATH] = GridCell.PATH

        if self.base is not None:
            y, x = self.base.pos 
            self.grid[y, x, GridCell.BASE] = GridCell.BASE


    def _action_to_coordinate(self, n, action):
        """
            Maps an action in [0, n) to a coordinate on an nxn grid
            Args:
                n: the size of the nxn grid
                action: the action in [0, n) to take
        """
        # divmod(action, n) converts the flat action index into a 2D grid coordinate (y, x).
        return divmod(action, n)

    
    def _render_frame(self):
        """ Draws current state to pygame display """
        if self.render_mode != "human":
            return
        
        cell_size = 80
        sidebar_width = 200
        window_size = self.size * cell_size

        if self.window is None: # initialize
            pygame.init()
            self.window = pygame.display.set_mode((window_size + sidebar_width, window_size))
            pygame.display.set_caption(ENV_NAME)
            self.font = pygame.font.SysFont("consolas", 16)

        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))

        color_map = {
            GridCell.TOWER: (69, 203, 133),  # tower
            GridCell.ENEMY: (200, 0, 0),     # enemy
            GridCell.BASE: (0, 255, 180),    # base 
            GridCell.PATH: (150, 150, 150),  # path
            GridCell.EMPTY: (240, 240, 240), # empty
        }

        # Draw the grid
        for y in range(self.size):
            for x in range(self.size):

                if self.grid[y, x, GridCell.BASE] == GridCell.BASE: # draw base
                    # Color base based on health (green -> yellow -> red)
                    health_ratio = self.base.health_ratio()
                    if health_ratio > 0.66:
                        base_color = (0, 100, 200)  # Blue when healthy
                    elif health_ratio > 0.33:
                        base_color = (200, 150, 0)  # Orange when damaged
                    else:
                        base_color = (200, 0, 0)    # Red when critical
                    pygame.draw.rect(canvas, base_color, pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size))

                elif self.grid[y, x, GridCell.PATH] == GridCell.PATH: # draw path
                    pygame.draw.rect(canvas, color_map[GridCell.PATH], pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size))

                else: # draw empty square
                    pygame.draw.rect(canvas, color_map[GridCell.EMPTY], pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size))

                
                if self.grid[y, x, GridCell.ENEMY] == GridCell.ENEMY: # Draw enemy
                    cell_x = x * cell_size
                    cell_y = y * cell_size
                    color = color_map[GridCell.ENEMY] 
                    pygame.draw.line(canvas, color, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), 4) # Draw an X shape (two diagonal lines)
                    pygame.draw.line(canvas, color, (cell_x + cell_size, cell_y), (cell_x, cell_y + cell_size), 4)
                    # pygame.draw.rect(canvas, color_map[GridCell.ENEMY], pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size))
                
                # Different tower types
                if self.grid[y, x, GridCell.TOWER] == GridCell.TOWER: # Draw tower
                    center = (x*cell_size + cell_size//2, y*cell_size + cell_size//2)
                    tower = self.towers.get((y, x))

                    # Draw tower level & damage it can deal
                    if tower is not None:

                        # Tower indicator
                        pygame.draw.circle(
                            canvas,
                            tower.get_color(),
                            center,
                            cell_size//3
                        )

                        # Tower level
                        level_text = str(tower.level)
                        font_level = pygame.font.SysFont("consolas", 18, bold=True)
                        lvl_surface = font_level.render(level_text, True, (0, 0, 0))
                        lvl_rect = lvl_surface.get_rect(center=(center[0], center[1] - 4))
                        canvas.blit(lvl_surface, lvl_rect)

                        # Damage
                        dmg_text = f"{float(tower.damage):.2f}"
                        font_dmg = pygame.font.SysFont("consolas", 12, bold=False)
                        dmg_surface = font_dmg.render(dmg_text, True, (0, 0, 0))
                        dmg_rect = dmg_surface.get_rect(center=(center[0], center[1] + 12))
                        canvas.blit(dmg_surface, dmg_rect)
                    else:
                        print(f"Could not find tower")

        # Draw grid lines
        for i in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (i*cell_size, 0), (i*cell_size, window_size))
            pygame.draw.line(canvas, (0, 0, 0), (0, i*cell_size), (window_size, i*cell_size))

        # Draw base border (thick border to highlight it)
        by, bx = self.base.pos
        pygame.draw.rect(canvas, (255, 255, 255), 
                        pygame.Rect(bx*cell_size, by*cell_size, cell_size, cell_size), 
                        width=4)  # Thick white border

        # Draw base health text
        # Draw "BASE" label
        font_label = pygame.font.Font(None, 28)
        label_surface = font_label.render("BASE", True, (255, 255, 255))
        label_rect = label_surface.get_rect(center=(bx*cell_size + cell_size//2, by*cell_size + cell_size//2 - 15))
        canvas.blit(label_surface, label_rect)
        
        # Draw health ratio
        font_health = pygame.font.Font(None, 32)
        health_text = f"{self.base.health}/{self.base.original_health}"
        text_surface = font_health.render(health_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(bx*cell_size + cell_size//2, by*cell_size + cell_size//2 + 12))
        canvas.blit(text_surface, text_rect)

        # Draw grid on window
        self.window.blit(canvas, (0, 0))

        # Draw sidebar on the window
        sidebar_width = 200
        sidebar_x = self.size * cell_size
        sidebar_y = self.size * cell_size

        # Sidebar background
        pygame.draw.rect(self.window, (255, 255, 255), pygame.Rect(sidebar_x, 0, sidebar_width, sidebar_y))

        # Sidebar border
        pygame.draw.line(self.window, (100, 100, 100), (sidebar_x, 0), (sidebar_x, sidebar_y), 2)

        # Render text
        title = self.font.render("Tower Defense RL", True, (27, 32, 33))
        episode = self.font.render(f"Episode: {self.current_episode}", True, (27, 32, 33))
        wave = self.font.render(f"Wave: {self.wave_count + 1}", True, (27, 32, 33))
        reward = self.font.render(f"TReward: {self.total_reward:.2f}", True, (27, 32, 33))
        avg_reward = self.font.render(f"Avg Reward: {self.total_reward if self.current_episode == 0 else (self.total_reward / self.current_episode):.2f}", True, (27, 32, 33))

        # Draw Stats
        self.window.blit(title, (sidebar_x + 15, 20))
        self.window.blit(episode, (sidebar_x + 15, 60))
        self.window.blit(wave, (sidebar_x + 15, 90))
        self.window.blit(reward, (sidebar_x + 15, 120))
        self.window.blit(avg_reward, (sidebar_x + 15, 150))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
