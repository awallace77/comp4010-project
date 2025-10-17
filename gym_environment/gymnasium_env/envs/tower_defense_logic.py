import numpy as np

class TowerDefenseGame:
    GRID_SIZE = 5
    ENEMY_HP = 10
    RANGE = 1
    BASE_HEALTH = 10
    MAX_ENEMIES = 4
    SPAWN_TIMING = 2

    def __init__(self):
        self.path = self.s_path(self.GRID_SIZE)

    #lays down coordiates for an S shape based on grid size
    @staticmethod
    def s_path(n: int, gap: int = 2):
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

newGame = TowerDefenseGame()
print(newGame.path)