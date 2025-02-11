import random
import numpy as np
import pygame
import gym
import time

class MinesweeperEnv(gym.Env):
    def __init__(self, width=8, height=8, num_mines=10, use_dfs=False):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.board = np.zeros((width, height), dtype=int)
        self.mines = set()
        self.done = False
        self.action_space = gym.spaces.Discrete(width * height)
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(width, height), dtype=int)
        # 0-8: number of mines around, 9: mine, 10: unknown
        # observation is the board state
        self.action_mask = np.ones(width * height, dtype=bool)
        self.use_dfs = use_dfs
        self.reset()
        
        # pygame settings
        self.cell_size = 41
        self.screen_width = self.width * self.cell_size
        self.screen_height = self.height * self.cell_size
        self.screen = None
        self.colors = {
            'bg': (192, 192, 192), # background color: light gray
            'grid': (128, 128, 128), # grid line color: gray
            'unknown': (160, 160, 160), # unknown cell color: light gray
            'mine': (255, 0, 0), # mine color: red
            'numbers': [
                (192, 192, 192), # 0: light gray
                (0, 0, 255), # 1: blue
                (0, 128, 0), # 2: green
                (255, 0, 0), # 3: red
                (0, 0, 128), # 4: dark blue
                (128, 0, 0), # 5: dark red
                (0, 128, 128), # 6: cyan
                (0, 0, 0), # 7: black
                (128, 128, 128) # 8: gray
            ]
        }
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 36)
        
    def reset(self):
        self.board.fill(10)
        self.mines = set()
        self.action_mask.fill(True)
        
        # calculate safe positions
        center_x = self.width // 2
        center_y = self.height // 2
        
        # generate safe positions
        safe_positions = set()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x, new_y = center_x + dx, center_y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    safe_positions.add((new_x, new_y))
        
        # generate mines
        mines_placed = 0
        while mines_placed < self.num_mines:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if (x, y) not in self.mines and (x, y) not in safe_positions:
                self.mines.add((x, y))
                mines_placed += 1
        
        self.done = False
        return self.board.copy()
    
    def step(self, action):
        x, y = action // self.height, action % self.height
        
        assert self.action_mask[action], 'Invalid action'
        self.action_mask[action] = False
        
        if (x, y) in self.mines:
            # Game over
            self.board[x, y] = 9
            self.done = True
            return self.board.copy(), -(self.width * self.height), self.done, {}
        else:
            self.board[x, y] = self.count_mines(x, y)
            if self.board[x, y] == 0 and self.use_dfs:
                self._update_mask_dfs(x, y)
            if np.count_nonzero(self.board == 10) == self.num_mines:
                self.done = True
                return self.board.copy(), self.width * self.height, self.done, {}
            return self.board.copy(), 1, self.done, {}
    
    def _update_mask_dfs(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and 
                    0 <= new_y < self.height):
                    action = new_x * self.height + new_y
                    if self.action_mask[action]:
                        self.action_mask[action] = False
                        self.board[new_x, new_y] = self.count_mines(new_x, new_y)
                        if self.count_mines(new_x, new_y) == 0:
                            self._update_mask_dfs(new_x, new_y)
    def count_mines(self, x, y):
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                    if (x + dx, y + dy) in self.mines:
                        count += 1
        return count
                            
    def render(self, mode='print'):
        if mode == 'print':
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[x, y] == 10:
                        print('.', end=' ')
                    elif self.board[x, y] == 9:
                        print('*', end=' ')
                    else:
                        print(self.board[x, y], end=' ')
        elif mode == 'pygame':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption('Minesweeper')
            
            # fill in background color
            self.screen.fill(self.colors['bg'])
            
            # draw grid lines
            for y in range(self.height):
                for x in range(self.width):
                    rect = pygame.Rect(
                        x * self.cell_size, 
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    cell_value = self.board[x, y]
                    if cell_value == 10:
                        # unknown cell
                        pygame.draw.rect(self.screen, self.colors['unknown'], rect)
                    elif cell_value == 9:
                        pygame.draw.circle(
                            self.screen, 
                            self.colors['mine'],
                            (x * self.cell_size + self.cell_size//2, 
                            y * self.cell_size + self.cell_size//2),
                            self.cell_size//3
                        )
                    else:
                        text = self.font.render(
                            str(cell_value),
                            True,
                            self.colors['numbers'][cell_value] # index 0-7
                        )
                        text_rect = text.get_rect(
                            center=(x * self.cell_size + self.cell_size//2, 
                            y * self.cell_size + self.cell_size//2)
                        )
                        self.screen.blit(text, text_rect)
                    pygame.draw.rect(self.screen, self.colors['grid'], rect, 1)
            
            pygame.display.flip()
            
            time.sleep(1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
        return self.board.copy()
    
    def close(self):
        if self.screen is not None:
            pygame.quit()