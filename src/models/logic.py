# Add the src directory to Python path
import os, sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)
from env.minesweeper import MinesweeperEnv
from models.base import MinesweeperSolver
import numpy as np
import random
from config.ppo_config import PPOConfig
from utils.env_utils import make_env
import gymnasium as gym

class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells
        else:
            return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        else:
            return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            
class LogicSolver(MinesweeperSolver):
    """
    Minesweeper game player
    """

    def __init__(self, envs):
        # Set initial height and width
        self.num_envs, self.width, self.height = envs.observation_space.shape

        # Keep track of which cells have been clicked on for each environment
        self.moves_made = [set() for _ in range(envs.num_envs)]

        # Keep track of cells known to be safe or mines for each environment
        self.mines = [set() for _ in range(envs.num_envs)]
        self.safes = [set() for _ in range(envs.num_envs)]

        # List of sentences about the game known to be true for each environment
        self.knowledge = [[] for _ in range(envs.num_envs)]

    def mark_mine(self, env_idx, cell):
        self.mines[env_idx].add(cell)
        for sentence in self.knowledge[env_idx]:
            sentence.mark_mine(cell)

    def mark_safe(self, env_idx, cell):
        self.safes[env_idx].add(cell)
        for sentence in self.knowledge[env_idx]:
            sentence.mark_safe(cell)

    def add_knowledge(self, env_idx, cell, count):
        self.moves_made[env_idx].add(cell)
        self.mark_safe(env_idx, cell)

        newSentenceCells = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if 0 <= i < self.width and 0 <= j < self.height and (i, j) != cell:
                    if (i, j) in self.mines[env_idx]:
                        count -= 1
                    elif (i, j) not in self.safes[env_idx]:
                        newSentenceCells.add((i, j))

        newSentence = Sentence(newSentenceCells, count)
        self.knowledge[env_idx].append(newSentence)

        newMines = set()
        newSafes = set()

        for sentence in self.knowledge[env_idx]:
            for mine in sentence.known_mines():
                if mine not in self.mines[env_idx]:
                    newMines.add(mine)
            for safe in sentence.known_safes():
                if safe not in self.safes[env_idx]:
                    newSafes.add(safe)

        for mine in newMines:
            self.mark_mine(env_idx, mine)
        for safe in newSafes:
            self.mark_safe(env_idx, safe)

        newKnowledge = []
        for sentence1 in self.knowledge[env_idx]:
            for sentence2 in self.knowledge[env_idx]:
                if sentence1 != sentence2 and sentence1.cells.issubset(sentence2.cells):
                    newCells = sentence2.cells - sentence1.cells
                    newCount = sentence2.count - sentence1.count
                    newKnowledge.append(Sentence(newCells, newCount))

        for sentence in newKnowledge:
            if sentence not in self.knowledge[env_idx]:
                self.knowledge[env_idx].append(sentence)

        emptySentences = []
        for sentence in self.knowledge[env_idx]:
            if len(sentence.cells) == 0:
                emptySentences.append(sentence)

        for sentence in emptySentences:
            self.knowledge[env_idx].remove(sentence)

    def make_safe_move(self, env_idx):
        for cell in self.safes[env_idx]:
            if cell not in self.moves_made[env_idx]:
                return cell
        return None

    def make_random_move(self, env_idx):
        possibleMoves = []
        for i in range(self.width):
            for j in range(self.height):
                if (i, j) not in self.mines[env_idx] and (i, j) not in self.moves_made[env_idx]:
                    possibleMoves.append((i, j))

        if len(possibleMoves) == 0:
            return None
        else:
            return random.choice(possibleMoves)

    def get_actions(self, states):
        actions = []
        for env_idx, state in enumerate(states):
            if len(self.moves_made[env_idx]) == 0:
                center_x = self.width // 2
                center_y = self.height // 2
                self.moves_made[env_idx].add((center_x, center_y))
                actions.append(center_x * self.height + center_y)
            else:
                for x in range(self.width):
                    for y in range(self.height):
                        if (x, y) not in self.moves_made[env_idx] and state[x, y] != 10:
                            self.moves_made[env_idx].add((x, y))
                            self.add_knowledge(env_idx, (x, y), state[x, y])
                cell = self.make_safe_move(env_idx)
                if cell is None:
                    cell = self.make_random_move(env_idx)
                if cell is None:
                    # print(self.safes[env_idx])
                    # print(self.mines[env_idx])
                    # print(self.moves_made[env_idx])
                    print(f"env_idx: {env_idx}, no valid moves found.")
                assert cell is not None
                actions.append(cell[0] * self.height + cell[1])
        return actions

if __name__ == "__main__":
    env_config = {
        'width': 30,
        'height': 16,
        'num_mines': 99,
        'use_dfs': True,
    }
    run_name = "test"
    config = PPOConfig()
    envs = gym.vector.SyncVectorEnv(
        [make_env(config, config.seed + i, i, False, run_name) for i in range(config.num_envs)]
    )

    # env = MinesweeperEnv(env_config)
    solver = LogicSolver(envs)

    states, info = envs.reset()
    done = [False] * config.num_envs
    total_rewards = [0] * config.num_envs
    steps = [0] * config.num_envs

    for step in range(0, config.num_steps):
        actions = solver.get_actions(states)
        states, rewards, dones, terminates, infos = envs.step(actions)
        for i in range(config.num_envs):
            if "final_info" in infos:  # 检查是否有环境被重置
                if infos["final_info"][i] is not None:  # 该环境确实结束了
                    print(f"\nGame Over for env {i}! Total reward: {total_rewards[i]:.1f}")
                    # 只需重置 solver 的状态
                    solver.moves_made[i].clear()
                    solver.mines[i].clear()
                    solver.safes[i].clear()
                    solver.knowledge[i].clear()
                    # 重置计数器
                    total_rewards[i] = 0
                    steps[i] = 0
            else:  # 正常累加奖励
                total_rewards[i] += rewards[i]
                steps[i] += 1

    for i in range(config.num_envs):
        print(f"\nGame Over for env {i}! Total reward: {total_rewards[i]:.1f}")