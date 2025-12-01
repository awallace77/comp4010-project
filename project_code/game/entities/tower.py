from game.game_info import TowerInfo
import math

class Tower:
    """
        Defines a tower 
        Level 1 to 2: 10 kills
        Level 2 to 3: 20 more kills (30 total)
        Level 3 to 4: 30 more kills (60 total)
        Level 4 to 5: 40 more kills (100 total)
        Maximum level is 5
    """
    _id_counter = 1
    _id_map = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register subclass as soon as it is defined
        if cls not in Tower._id_map:
            Tower._id_map[cls] = Tower._id_counter
            Tower._id_counter += 1

    @classmethod
    def get_id(cls):
        return Tower._id_map[cls]

    @classmethod
    def get_num_types(cls):
        return len(cls._id_map)
    
    @classmethod
    def get_ids(cls):
        return cls._id_map
        
    def __init__(self, pos, health, damage, range, cost):
        self.health = health
        self.original_health = health
        self.damage = damage
        self.pos = pos
        self.range = range
        self.cost = cost

        self.level = 1
        self.kills = 0

    def attack(self, enemies):
        for enemy in enemies:
            ey, ex = enemy.pos
            y, x = self.pos
            # if(abs(ey - y) + abs(ex - x) <= self.range): # manhattan distance is within range

            # Same row or column and within range
            if(
                (ey == y and abs(ex-x) <= self.range) or 
                (ex == x and abs(ey-y) <= self.range)
            ):
                enemy.take_damage(self.damage)
                return [enemy]
        return []
    
    def move(self, pos):
        self.pos = pos

    def take_damage(self, damage):
        self.health -= damage

    def is_dead(self):
        return self.health <= 0
    
    def respawn(self):
        self.health = self.original_health

    def level_up(self):
        """Returns true if valid level occurred, false otherwise"""
        if self.level < TowerInfo.MAX_LEVEL:
            self.level += 1
            # Increment damage by LEVEL_DAMAGE_DELTA % of current damage
            self.damage += math.ceil(TowerInfo.LEVEL_DAMAGE_DELTA * self.damage)
            return True
        return False

    def killed_enemy(self):
        """Returns true if leveled up after kill, false otherwise"""
        self.kills += 1
        if self.kills > self.level * TowerInfo.LEVEL_KILLS_DELTA:
            return self.level_up()
        return False
    
    def get_color(self):
        return (0, 0, 0) # default tower color


class SingleTargetTower(Tower):
    """
        Single Target Tower that deals damage to a single enemy
    """
    def __init__(self, 
                 pos, 
                 health=TowerInfo.SINGLE_TARGET_HEALTH, 
                 damage=TowerInfo.SINGLE_TARGET_DAMAGE, 
                 range=TowerInfo.SINGLE_TARGET_RANGE, 
                 cost=TowerInfo.SINGLE_TARGET_COST):
        super().__init__(pos, health, damage, range, cost)

    def get_color(self):
        return (69, 203, 133) 


class AoETower(Tower):
    """
        Area of Effect (AoE) Tower that can damage enemies in its vicinity
    """
    def __init__(self, 
                 pos, 
                 health=TowerInfo.AOE_HEALTH, 
                 damage=TowerInfo.AOE_DAMAGE, 
                 range=TowerInfo.AOE_RANGE, 
                 cost=TowerInfo.AOE_COST):
        super().__init__(pos, health, damage, range, cost)

    def attack(self, enemies):
        affected = []
        y, x = self.pos
        for e in enemies:
            ey, ex = e.pos
         
            # Deal damage to multiple enemies in a radius around the tower 
            if (ey == y and abs(ex-x) <= self.range) or (ex == x and abs(ey-y) <= self.range):
                e.take_damage(self.damage)
                affected.append(e)

        return affected
    
    def get_color(self):
        return (255, 140, 0)


