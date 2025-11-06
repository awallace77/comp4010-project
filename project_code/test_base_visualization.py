"""
Test script to visualize the base at different health levels
"""
import pygame
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tower_defense_world import TowerDefenseWorld
import time

def test_base_visualization():
    """Show the base at different health levels"""
    
    env = TowerDefenseWorld(render_mode='human', num_enemies=3)
    state, info = env.reset()
    
    print("=== BASE VISUALIZATION TEST ===")
    print(f"Base location: {env.base}")
    print(f"Starting health: {env.base_health}/{env.base_health_max}")
    print()
    
    # Show initial state
    print("1. Showing base at full health (blue)...")
    env.render()
    time.sleep(2)
    
    # Damage base to medium health
    env.base_health = 12
    print(f"2. Base damaged to {env.base_health}/{env.base_health_max} (orange)...")
    env.render()
    time.sleep(2)
    
    # Damage base to low health
    env.base_health = 5
    print(f"3. Base critical at {env.base_health}/{env.base_health_max} (red)...")
    env.render()
    time.sleep(2)
    
    # Restore health
    env.base_health = env.base_health_max
    print(f"4. Base restored to {env.base_health}/{env.base_health_max} (blue)...")
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
