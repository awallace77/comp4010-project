from dataclasses import dataclass
"""
    game_info.py
    Stores all the default info for the game
"""

@dataclass(frozen=True)
class BaseInfo:
    MAX_HEALTH: int = 40
    HEALTH: int = 20

@dataclass(frozen=True)
class EnemyInfo:
    MAX_HEALTH: int = 26
    MAX_DAMAGE: int = 6
    # MAX_LEAK: int = 20
    
    HEALTH: int = 13
    DAMAGE: int = 3
    # LEAK: int = 10

@dataclass(frozen=True)
class TowerInfo:
    MAX_HEALTH: int = 56
    MAX_DAMAGE: int = 8
    MAX_COST: int = 130
    
    SINGLE_TARGET_HEALTH: int = 28
    SINGLE_TARGET_DAMAGE: int = 4
    SINGLE_TARGET_RANGE: int = 2
    SINGLE_TARGET_COST: int = 50

    AOE_HEALTH: int = 28
    AOE_DAMAGE: int = 4
    AOE_RANGE: int = 2
    AOE_COST: int = 65


