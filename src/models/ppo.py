from src import MinesweeperSolver
import numpy as np

class PPO(MinesweeperSolver):
    def __init__(self, env):
        super().__init__(env)
        self.width = env.width
        self.height = env.height
        self.num_mines = env.num_mines