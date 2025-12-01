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
    IDLE_P: float = 0.1 # probability of not moving is X% of the time
    # LEAK: int = 10
    KILL_PAYOUT: int = 10 # the "coins" received after a kill

@dataclass(frozen=True)
class TowerInfo:
    MAX_HEALTH: int = 56
    MAX_DAMAGE: int = 8
    MAX_COST: int = 130
    MAX_LEVEL: int = 5
    LEVEL_KILLS_DELTA: int = 10 # number of kills required between levels
    LEVEL_DAMAGE_DELTA: float = 0.15 # percentage to increase damage by between levels
    
    SINGLE_TARGET_HEALTH: int = 28
    SINGLE_TARGET_DAMAGE: int = 4
    SINGLE_TARGET_RANGE: int = 3
    SINGLE_TARGET_COST: int = 50

    AOE_HEALTH: int = 28
    AOE_DAMAGE: int = 1.5
    AOE_RANGE: int = 2
    AOE_COST: int = 65

@dataclass(frozen=True)
class BudgetInfo:
    MAX_BUDGET: int = 10000
    BUDGET: int = 150 # the default budget




