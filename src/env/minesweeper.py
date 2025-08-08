import random
import numpy as np
import pygame
import gymnasium as gym
import time
import os, tomli
from models.logic import LogicSolver

def load_rewards_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config/rewards.toml')
    config_path = os.path.abspath(config_path)
    with open(config_path, 'rb') as f:
        rewards = tomli.load(f)
    return rewards["rewards"]

class MinesweeperEnv(gym.Env):
    def __init__(self, config):
        width = config.get('width', 8)
        height = config.get('height', 8)
        num_mines = config.get('num_mines', 10)
        use_dfs = config.get('use_dfs', True)
        self.num_envs = config.get('num_envs', 1)
        
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
        self.np_random = None
        self.current_seed = 0
        
        # variable to track information
        self.info = {
            'is_success': False,
            'is_game_over': False,
            'action_mask': self.get_action_mask(),
            'revealed_cells': 0,
        }

        # rewards configuration and relevant variables
        self.rewards_config = load_rewards_config() # a dict with coefficients
        self._last_uncertainty = None # uncertainty of the last action
        class _DummyObsSpace:
            def __init__(self, w, h): self.shape = (1, w, h)
        class _DummyEnvs:
            def __init__(self, w, h):
                self.observation_space = _DummyObsSpace(w, h)
                self.num_envs = 1

        self._dummy_envs = _DummyEnvs(self.width, self.height)
        self.solver = LogicSolver(self._dummy_envs)  # 持久化
        self._solver_env_idx = 0
        self._fed_cells = set()  # 记录已经喂过的数字格，避免重复 add_knowledge

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
        
    def seed(self, seed=None):
        """Set seed for random number generators"""
        self.current_seed = seed
        
    def reset(self, seed = None, **kwargs):
        if seed is None:
            self.current_seed = self.current_seed + self.num_envs
            seed = self.current_seed

        assert seed is not None, 'Seed must be set'
        seed = seed % 2**32
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        
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
            x = self.np_random.integers(0, self.width)
            y = self.np_random.integers(0, self.height)
            if (x, y) not in self.mines and (x, y) not in safe_positions:
                self.mines.add((x, y))
                mines_placed += 1
        
        self.done = False

        self.solver = LogicSolver(self._dummy_envs)   # 新的空 solver
        self._fed_cells.clear()
        U0 = self._compute_uncertainty()
        self._last_uncertainty = U0

        return self.board.copy(), {'action_mask': self.get_action_mask()}
    
    def step(self, action):
        x, y = action // self.height, action % self.height

        assert self.action_mask[action], 'Invalid action'

        # ====== 动作前情报 ======
        U_before = self._last_uncertainty if self._last_uncertainty is not None else self._compute_uncertainty()
        deducible_safe, deducible_mines = self._deducible_safe_and_mines()
        no_deducible_move = (len(deducible_safe) == 0)  # 你没有“插旗”动作，地雷可推断在这版用不上

        self.action_mask[action] = False
        last_revealed = np.count_nonzero(self.board != 10)

        # ====== 踩雷分支 ======
        if (x, y) in self.mines:
            self.board[x, y] = 9
            self.done = True
            self.info.update({
                "is_game_over": True,
                "revealed_cells": np.count_nonzero(self.board != 10),
                "action_mask": self.get_action_mask(),
            })
            reward = self.rewards_config["lose"]
            self._last_uncertainty = 0.0
            obs = self.board.copy()
            return obs, reward, self.done, False, self.info

        # ====== 安全分支：正常翻开 + DFS ======
        self.board[x, y] = self.count_mines(x, y)
        self._solver_feed_cell(x, y, self.board[x, y])
        if self.board[x, y] == 0 and self.use_dfs:
            self._update_mask_dfs(x, y)

        if np.count_nonzero(self.board == 10) == self.num_mines:
            self.done = True
            self.info.update({
                "is_success": True,
                "revealed_cells": np.count_nonzero(self.board != 10),
                "action_mask": self.get_action_mask(),
            })
            obs = self.board.copy()
            reward = self.rewards_config["win"]
            self._last_uncertainty = 0.0
            return obs, reward, self.done, False, self.info

        obs = self.board.copy()
        self.info.update({
            "is_success": False,
            "is_game_over": False,
            "revealed_cells": np.count_nonzero(self.board != 10),
            "action_mask": self.get_action_mask(),
        })
        revealed_increase = np.count_nonzero(self.board != 10) - last_revealed
        assert revealed_increase > 0, "Revealed cells should increase"

        U_after = self._compute_uncertainty()
        self._last_uncertainty = U_after

        r_info = self.rewards_config["info_gain"] * max(0.0, U_before - U_after)
        denom_safe = max(1, (self.width * self.height - self.num_mines))
        r_flood = self.rewards_config["flood"] * (revealed_increase / denom_safe)
        r_logic = self.rewards_config["logic"] * (revealed_increase / denom_safe) if (x, y) in deducible_safe else 0.0
        r_guess = self.rewards_config["eta_guess"] * (revealed_increase / denom_safe) if no_deducible_move else 0.0

        reward = r_info + r_flood + r_logic + r_guess
        return obs, reward, self.done, False, self.info

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
                        val = self.count_mines(new_x, new_y)
                        self.board[new_x, new_y] = val
                        self._solver_feed_cell(new_x, new_y, val)
                        if val == 0:
                            self._update_mask_dfs(new_x, new_y)

    def _neighbors(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    yield new_x, new_y

    def _compute_uncertainty(self):
        hidden_coords = list(zip(*np.where(self.board == 10)))
        hidden = len(hidden_coords)
        if hidden == 0:
            return 0.0

        total_mines = self.num_mines
        revealed_mines = sum(1 for (x,y) in self.mines if self.board[x,y] == 9)
        remaining_mines = max(0, total_mines - revealed_mines)

        p_global = remaining_mines / hidden if hidden > 0 else 0.0

        safe, mines = self._deducible_safe_and_mines()

        def H(p):
            if p <= 0.0 or p >= 1.0:
                return 0.0
            return -(p*np.log(p) + (1-p)*np.log(1-p))
        U = 0.0
        for c in hidden_coords:
            if c in safe:
                p = 0.0
            elif c in mines:
                p = 1.0
            else:
                p = p_global
            U += H(p)
        return float(U)
    
    def _solver_feed_cell(self, x, y, val):
        """
        把新揭开的数字格 (x,y,val) 喂给持久 solver。
        只喂 0-8；9(踩雷) 不喂（游戏结束无意义）。
        """
        if val in range(0, 9):
            if (x, y) not in self._fed_cells:
                self.solver.add_knowledge(self._solver_env_idx, (x, y), int(val))
                self._fed_cells.add((x, y))

    def _deducible_safe_and_mines(self):
        env_idx = self._solver_env_idx
        safe = {(x,y) for (x,y) in self.solver.safes[env_idx] if self.board[x,y] == 10}
        mines = {(x,y) for (x,y) in self.solver.mines[env_idx] if self.board[x,y] == 10}
        return safe, mines
    
    def count_mines(self, x, y):
        count = 0
        for neighbor in self._neighbors(x, y):
            if neighbor in self.mines:
                count += 1
        return count
                            
    def render(self, mode="rgb_array"):
        """
        mode: 'rgb_array' | 'pygame' | 'human' | 'print'
        - rgb_array: 离屏渲染，返回(H, W, 3) numpy
        - pygame/human: 开窗渲染，同时返回帧；若窗口被关，返回 None
        - print: 控制台打印棋盘
        """
        import pygame, os
        mode = (mode or getattr(self, "render_mode", "rgb_array")).lower()

        # ---------- helpers ----------
        def _ensure_pygame():
            if not pygame.get_init():
                pygame.init()
            if not pygame.font.get_init():
                pygame.font.init()
            # 确保有字体
            if getattr(self, "font", None) is None:
                # 字体大小给点边距
                size = max(12, self.cell_size - 4)
                self.font = pygame.font.SysFont(None, size)
            # 控帧器
            if getattr(self, "clock", None) is None:
                self.clock = pygame.time.Clock()

        def _draw(surface):
            # 背景
            surface.fill(self.colors['bg'])
            # 网格
            for y in range(self.height):
                for x in range(self.width):
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    cell_value = int(self.board[x, y])

                    if cell_value == 10:
                        # 未知
                        pygame.draw.rect(surface, self.colors['unknown'], rect)
                    elif cell_value == 9:
                        # 地雷
                        pygame.draw.circle(
                            surface,
                            self.colors['mine'],
                            (x * self.cell_size + self.cell_size // 2,
                            y * self.cell_size + self.cell_size // 2),
                            max(2, self.cell_size // 3)
                        )
                    else:
                        # 数字；0 就别画字了，直接空白
                        if cell_value != 0:
                            color = self.colors['numbers'][cell_value]
                            txt = self.font.render(str(cell_value), True, color)
                            txt_rect = txt.get_rect(center=(
                                x * self.cell_size + self.cell_size // 2,
                                y * self.cell_size + self.cell_size // 2
                            ))
                            surface.blit(txt, txt_rect)

                    # 画边框
                    pygame.draw.rect(surface, self.colors['grid'], rect, 1)

        # ---------- modes ----------
        if mode == 'print':
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    v = int(self.board[x, y])
                    row.append('.' if v == 10 else ('*' if v == 9 else str(v)))
                print(' '.join(row))
            return self.board.copy()

        if mode in ('rgb_array',):
            _ensure_pygame()
            # 离屏 surface
            w, h = self.screen_width, self.screen_height
            surf = pygame.Surface((w, h))
            _draw(surf)
            frame = pygame.surfarray.array3d(surf).swapaxes(0, 1)  # (H, W, 3)
            # 控制渲染帧率（可选）：默认 30fps
            self.clock.tick(getattr(self, "render_fps", 30))
            return frame

        if mode in ('human', 'pygame'):
            _ensure_pygame()
            # 若 display 被关/未建，重建窗口
            if not pygame.display.get_init() or getattr(self, "screen", None) is None:
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption('Minesweeper')

            # 处理事件；用户点 X 就优雅退出窗口，并返回 None
            quit_now = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    quit_now = True
            if quit_now:
                pygame.display.quit()
                self.screen = None
                return None

            if self.screen is None or not pygame.display.get_init():
                return None
            try:
                _draw(self.screen)
            except pygame.error:
                # Surface 已经 quit，不能再画
                self.screen = None
                return None
            pygame.display.flip()
            frame = pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
            self.clock.tick(getattr(self, "render_fps", 30))
            return frame

        raise ValueError(f"Unsupported render mode: {mode}")

    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            
    def get_action_mask(self):
        return self.action_mask.copy()