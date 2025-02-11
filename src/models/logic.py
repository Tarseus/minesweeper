from src import MinesweeperSolver
import numpy as np
import random

class Sentence:
    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count
        
    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count
    
    def __str__(self):
        return f"{self.cells} = {self.count}"
    
    def known_mines(self):
        if len(self.cells) == self.count and self.count != 0:
            return self.cells.copy()
        else:
            return set()
        
    def known_safes(self):
        if self.count == 0:
            return self.cells.copy()
        else:
            return set()
        
    def mark_mine(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1
            
    def mark_safe(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)
            
class LogicSolver(MinesweeperSolver):
    def __init__(self, env):
        super().__init__(env)
        self.width = env.width
        self.height = env.height
        self.num_mines = env.num_mines
        self.moves_made = set()
        self.knowledge = []
        self.mines = set()
        self.safe = set()

    def mark_mine(self, cell):
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)
            
    def mark_safe(self, cell):
        self.safe.add(cell)
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

        # Mark the cell as a move that has been made, and mark as safe:
        self.moves_made.add(cell)
        self.mark_safe(cell)

        # Create set to store undecided cells for KB:
        new_sentence_cells = set()

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # If cells are already safe, ignore them:
                if (i, j) in self.safe:
                    continue

                # If cells are known to be mines, reduce count by 1 and ignore them:
                if (i, j) in self.mines:
                    count = count - 1
                    continue

                # Otherwise add them to sentence if they are in the game board:
                if 0 <= i < self.height and 0 <= j < self.width:
                    new_sentence_cells.add((i, j))

        # Add the new sentence to the AI's Knowledge Base:
        self.knowledge.append(Sentence(new_sentence_cells, count))

        # Iteratively mark guaranteed mines and safes, and infer new knowledge:
        knowledge_changed = True

        while knowledge_changed:
            knowledge_changed = False

            safes = set()
            mines = set()

            # Get set of safe spaces and mines from KB
            for sentence in self.knowledge:
                safes = safes.union(sentence.known_safes())
                mines = mines.union(sentence.known_mines())

            # Mark any safe spaces or mines:
            if safes:
                knowledge_changed = True
                for safe in safes:
                    self.mark_safe(safe)
            if mines:
                knowledge_changed = True
                for mine in mines:
                    self.mark_mine(mine)

            # Remove any empty sentences from knowledge base:
            empty = Sentence(set(), 0)

            self.knowledge[:] = [x for x in self.knowledge if x != empty]

            # Try to infer new sentences from the current ones:
            for sentence_1 in self.knowledge:
                for sentence_2 in self.knowledge:

                    # Ignore when sentences are identical
                    if sentence_1.cells == sentence_2.cells:
                        continue

                    if sentence_1.cells == set() and sentence_1.count > 0:
                        print('Error - sentence with no cells and count created')
                        raise ValueError

                    # Create a new sentence if 1 is subset of 2, and not in KB:
                    if sentence_1.cells.issubset(sentence_2.cells):
                        new_sentence_cells = sentence_2.cells - sentence_1.cells
                        new_sentence_count = sentence_2.count - sentence_1.count

                        new_sentence = Sentence(new_sentence_cells, new_sentence_count)

                        # Add to knowledge if not already in KB:
                        if new_sentence not in self.knowledge:
                            knowledge_changed = True
                            self.knowledge.append(new_sentence)

        # print('Current AI KB length: ', len(self.knowledge))
        # print('Known Mines: ', self.mines)
        # print('Safe Moves Remaining: ', self.safe - self.moves_made)
        # print('====================================================')

        
    def get_neighbors(self, cell):
        neighbors = set()
        x, y = cell
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= x + i < self.width and 0 <= y + j < self.height:
                    neighbors.add((x + i, y + j))
        return neighbors
    
    def update_knowledge(self):
        for sentence in self.knowledge:
            known_mines = sentence.known_mines()
            known_safes = sentence.known_safes()
            for mine in known_mines:
                self.mark_mine(mine)
            for safe in known_safes:
                self.mark_safe(safe)
        self.knowledge = [sentence for sentence in self.knowledge if sentence.cells]
    
    def get_action(self, env):
        valid_actions = np.where(self.env.action_mask)[0]
        
        if len(valid_actions) == 0:
            return None
            
        if np.all(env.board == 10):
            center_x = self.width // 2
            center_y = self.height // 2
            self.add_knowledge((center_x, center_y), self.env.count_mines(center_x, center_y))
            return center_x * self.height + center_y
        safe_moves = self.safe - self.moves_made
        if safe_moves:
            print("Safe move")
            # return random.choice(list(safe_moves))
            x, y = random.choice(list(safe_moves))
            self.add_knowledge((x, y), self.env.count_mines(x, y))
            return x * self.height + y
        
        print("Random move")
        action = random.choice(list(valid_actions))
        x, y = action // self.height, action % self.height
        self.add_knowledge((x, y), self.env.count_mines(x, y))
        return action