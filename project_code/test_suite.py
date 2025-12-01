"""
Test script to visualize the base at different health levels
"""
from envs.tower_defense_env import TowerDefenseEnv
from game.entities.tower import SingleTargetTower
from game.game_info import TowerInfo
import time
import math

def test_base_visualization():
    """Show the base at different health levels"""
    
    env = TowerDefenseEnv(render_mode='human', num_enemies=3)
    state, info = env.reset()
    
    print("=== BASE VISUALIZATION TEST ===")
    print(f"Base location: {env.base}")
    print(f"Starting health: {env.base.health}/{env.base.original_health}")
    print()
    
    # Show initial state
    print("1. Showing base at full health (blue)...")
    env.render()
    time.sleep(2)
    
    # Damage base to medium health
    env.base.health = 12
    print(f"2. Base damaged to {env.base.health}/{env.base.original_health} (orange)...")
    env.render()
    time.sleep(2)
    
    # Damage base to low health
    env.base.health = 5
    print(f"3. Base critical at {env.base.health}/{env.base.original_health} (red)...")
    env.render()
    time.sleep(2)
    
    # Restore health
    env.base.respawn()
    print(f"4. Base restored to {env.base.health}/{env.base.original_health} (blue)...")
    env.render()
    time.sleep(2)
    
    env.close()
    print("\n✓ Visualization test complete!")
    print("\nBase color indicators:")
    print("  • BLUE (66%+): Healthy base")
    print("  • ORANGE (33-66%): Damaged base")
    print("  • RED (<33%): Critical base")

def test_tower_visualization():    
    """Show the tower at different levels"""
    
    env = TowerDefenseEnv(render_mode='human', num_enemies=3)
    state, info = env.reset()
    n = env.size
    pos = n - math.floor(n/2), n - math.floor(n/2)
    tower = SingleTargetTower(pos=pos)
    env.add_tower_to_grid(tower)
    
    print("=== TOWER VISUALIZATION TEST ===")
    print(f"Tower location: {tower.pos}")
    print(f"Starting level: {tower.level}")
    print()
    
    # Show initial state
    for i in range(TowerInfo.MAX_LEVEL):
        print(f"Showing tower at level {i}, damage is {tower.damage}")
        env.render()
        time.sleep(2)
        tower.level_up()
    
    env.close()
    print("\n✓ Visualization test complete!")
    print("\nTower level up indicators:")
    print("  • 1: Level 1")
    print("  • 2: Level 2")
    print("  • 3: Level 3")
    print("  • 4: Level 4")
    print("  • 5: Level 5")

def test_enemy_spawning():
    env = TowerDefenseEnv(render_mode="human")
    state, info = env.reset()

    current_wave = info.get("wave", 1)
    print("\n===WAVE SPAWNING VISUALIZATION TEST===")
    print(f"Starting at wave {current_wave}")

    max_steps = 20000
    steps = 0

    while steps < max_steps:
        # Take a random action to advance the game so waves spawn.
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        steps += 1

        new_wave = info.get("wave", current_wave)

        # Detect when a new wave starts
        if new_wave != current_wave:
            current_wave = new_wave
            print(f"Reached wave {current_wave}")

        #Stop if episode is over
        if terminated or truncated:
            print("Episode finished early.")
            break

    env.close()


if __name__ == "__main__":
    """
        TEST SUITE: Add test functions here
    """
    test_base_visualization()
    test_tower_visualization()
    test_enemy_spawning()
