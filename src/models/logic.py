# Add the src directory to Python path
import os, sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)
from env.minesweeper import MinesweeperEnv
from models.base import MinesweeperSolver
import numpy as np
import random

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

    def __init__(self, env):

        # Set initial height and width
        self.height = env.height
        self.width = env.width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """ 
        self.moves_made.add(cell)
        self.mark_safe(cell)

        # start new sentence
        newSentenceCells = set()

        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if 0 <= i < self.width and 0 <= j < self.height and (i,j) != cell:
                    if (i,j) in self.mines:
                        count -= 1
                    elif (i,j) not in self.safes:
                        newSentenceCells.add((i,j))
        
        newSentence = Sentence(newSentenceCells, count)
        self.knowledge.append(newSentence)

        # mark additional cells as safe or mines based on any of the sentences
        newMines = set()
        newSafes = set()

        for sentence in self.knowledge:
            for mine in sentence.known_mines():
                if mine not in self.mines:
                    newMines.add(mine)
            for safe in sentence.known_safes():
                if safe not in self.safes:
                    newSafes.add(safe)

        for mine in newMines:
            self.mark_mine(mine)
        for safe in newSafes:
            self.mark_safe(safe)

        # add new sentences to the AI's knowledge based on any of the sentences
        newKnowledge = []
        for sentence1 in self.knowledge:
            for sentence2 in self.knowledge:
                if sentence1 != sentence2 and sentence1.cells.issubset(sentence2.cells):
                    newCells = sentence2.cells - sentence1.cells
                    newCount = sentence2.count - sentence1.count
                    newKnowledge.append(Sentence(newCells, newCount))

        for sentence in newKnowledge:
            if sentence not in self.knowledge:
                self.knowledge.append(sentence)

        # remove empty sentences
        emptySentences = []
        for sentence in self.knowledge:
            if len(sentence.cells) == 0:
                emptySentences.append(sentence)

        for sentence in emptySentences:
            self.knowledge.remove(sentence)
            

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell   
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        possibleMoves = []

        for i in range(self.width):
            for j in range(self.height):
                if (i,j) not in self.mines and (i,j) not in self.moves_made:
                    possibleMoves.append((i,j))
        
        if len(possibleMoves) == 0:
            return None
        else:
            return random.choice(possibleMoves)
        
    def get_action(self, env):
        
        if len(self.moves_made) == 0:
            center_x = self.width // 2
            center_y = self.height // 2
            self.add_knowledge((center_x, center_y), env.count_mines(center_x, center_y))
            return center_x * self.height + center_y
        else:
            cell = self.make_safe_move()
            if cell is not None:
                self.add_knowledge(cell, env.count_mines(cell[0], cell[1]))
                return cell[0] * self.height + cell[1]
            else:
                cell = self.make_random_move()
                self.add_knowledge(cell, env.count_mines(cell[0], cell[1]))
                return cell[0] * self.height + cell[1]
    
if __name__ == "__main__":
    env_config = {
        'width': 8,
        'height': 8,
        'num_mines': 10,
        'use_dfs': False
    }
    env = MinesweeperEnv(env_config)
    solver = LogicSolver(env)
    
    # 运行一个回合
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = solver.get_action(env)
        if action is None:
            break
            
        state, reward, done, t, _ = env.step(action)
        total_reward += reward
        
        env.render(mode='pygame')
        
    print(f"\nGame Over! Total reward: {total_reward}")