from game.game_info import BaseInfo
class Base:
    """
        Defines the base
    """
    def __init__(self, pos, health=BaseInfo.HEALTH):
        self.pos = pos
        self.health = health
        self.original_health = health

    def take_damage(self, damage):
        self.health -= damage

    def is_dead(self):
        return self.health <= 0
    
    def respawn(self):
        self.health = self.original_health

    def health_ratio(self):
        """Returns current health / original health"""
        return self.health / self.original_health

