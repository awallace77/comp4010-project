from game.game_info import TowerInfo

class Tower:
    """
        Defines a tower 
    """
    def __init__(self, pos, health, damage, range, cost):
        self.health = health
        self.original_health = health
        self.damage = damage
        self.pos = pos
        self.range = range
        self.cost = cost

    def attack(self, enemies):
        for enemy in enemies:
            ey, ex = enemy.pos
            y, x = self.pos
            # if(abs(ey - y) + abs(ex - x) <= self.range): # manhattan distance is within range

            # Same row or column and within range
            if(
                (ey == y and abs(ey-y) <= self.range) or 
                (ex == x and abs(ex-x) <= self.range)
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
        for e in enemies:
            ey, ex = e.pos
            y, x = self.pos

            # if(abs(ey - y) + abs(ex - x) <= self.range): # manhattan distance is within range
         
            # Deal damage to multiple enemies (Same row or column and within range)
            if(
                (ey == y and abs(ey-y) <= self.range) or 
                (ex == x and abs(ex-x) <= self.range)
            ):
                affected.append(e)
        for enemy in affected:
            enemy.take_damage(self.damage)
        return affected


