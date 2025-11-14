from game.game_info import EnemyInfo
import random

class Enemy:
    """
        Defines an enemy
    """
    def __init__(self, pos, path, health=EnemyInfo.HEALTH, damage=EnemyInfo.DAMAGE):
        self.pos = pos
        self.path = path # assumed valid path to base
        self.health = health
        self.original_health = health
        self.damage = damage

    def attack(self, towers):
        for tower in towers:
            if tower.pos == self.pos:
                tower.take_damage(self.damage)
                return [tower]
        return []
    
    def move(self):

        # Some of the time, idle (don't move)
        p = random.random() 
        if p < EnemyInfo.IDLE_P:
            return False
        
        next_idx = self.path.index(self.pos) + 1
        if next_idx < len(self.path):
            self.pos = self.path[next_idx]
            return False
        return True # enemy reached base
    
    def next_pos(self):
        next_idx = self.path.index(self.pos) + 1
        return self.path[next_idx]

    def take_damage(self, damage):
        self.health -= damage

    def is_dead(self):
        return self.health <= 0
    
    def respawn(self):
        self.health = self.original_health

# Can extend with different types of enemies if needed