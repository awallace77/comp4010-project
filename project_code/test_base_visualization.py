"""
Test script to visualize the base at different health levels
"""
import pygame
import sys
import os

# Add parent directory to path for imports
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.tower_defense_env import TowerDefenseEnv
import time

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

if __name__ == "__main__":
    test_base_visualization()
