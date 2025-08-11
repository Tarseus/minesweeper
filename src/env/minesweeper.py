import random
import numpy as np
import pygame
import gymnasium as gym
import time
import os, tomli
from src.algo.logic import LogicSolver

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
        self.safe_center = config.get('safe_center', False)
        self.first_click_safe = config.get('first_click_safe', True)
        self.phase = config.get('phase', "test")
        
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
        self.current_seed = config.get('seed', 0)
        self._true_board = None
        
        # variable to track information
        self.info = {
            'is_success': False,
            'is_game_over': False,
            'action_mask': self.get_action_mask(),
            'revealed_cells': 0,
            'full_board': None,  # will be set in reset()
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
        
    def seed(self, seed=None):
        """Set seed for random number generators"""
        self.current_seed = seed
        
    def reset(self, seed = None, safe_move = None, **kwargs):
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
        if safe_move is not None:
            safe_positions.add(safe_move)
        if self.safe_center:
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
        
        self._true_board = self._build_true_board()
        
        self.done = False

        self.solver = LogicSolver(self._dummy_envs)   # 新的空 solver
        self._fed_cells.clear()
        U0 = self._compute_uncertainty()
        self._last_uncertainty = U0

        self.info.update({
            "action_mask": self.get_action_mask(),
            "full_board": self._true_board.copy(),
        })

        return self.board.copy(), self.info
    
    def step(self, action):
        x, y = action // self.height, action % self.height

        assert self.action_mask[action], 'Invalid action'
                
        if np.all(self.board == 10) and self.first_click_safe: # 第一次动作
            if (x, y) in self.mines:
                self.reset(seed=self.current_seed, safe_move = (x, y) ) # 重新开局
                assert (x, y) not in self.mines, "First click should not be a mine"

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
    
    def count_mines(self, x, y):
        count = 0
        for neighbor in self._neighbors(x, y):
            if neighbor in self.mines:
                count += 1
        return count
                            
    def render(self, mode="rgb_array", probs=None, action=None,
           show_prob_text=True, number_text_scale=0.50, prob_text_scale=0.28,
           include_cbar=True):
        """
        新增:
        - number_text_scale: 左侧棋盘数字字号比例 (相对 cell_size)
        - prob_text_scale:   右侧热力图上概率文字字号比例
        - include_cbar:      是否在右侧加 colorbar
        """
        import numpy as np
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.patches import Rectangle, Circle
        import matplotlib.ticker as mticker

        mode = (mode or getattr(self, "render_mode", "rgb_array")).lower()

        W_cell = self.cell_size * self.width
        H_cell = self.cell_size * self.height
        gap = max(4, self.cell_size // 2)  # 左右面板间距
        cbar_pad = max(4, self.cell_size // 4)   # 右图与 colorbar 间距
        cbar_w = max(8, int(self.cell_size * 0.5))  # colorbar 宽度（像素）

        def _rgb255_to_1(col):
            return tuple(np.array(col, dtype=float) / 255.0)

        # --------- 解析 probs -> heat (H,W) ---------
        heat = None
        if probs is not None:
            p = np.asarray(probs)
            if p.ndim == 1:
                if p.size != self.width * self.height:
                    raise ValueError(f"Flat probs 大小应为 {self.width*self.height}，但得到 {p.size}")
                grid_x_y = p.reshape((self.width, self.height))  # (W,H)
                heat = grid_x_y.T                                 # (H,W)
            elif p.ndim == 2:
                if p.shape == (self.width, self.height):
                    heat = p.T
                elif p.shape == (self.height, self.width):
                    heat = p
                else:
                    raise ValueError(f"二维 probs 形状需为 (W,H) 或 (H,W)，但得到 {p.shape}")
            else:
                raise ValueError("probs 应为 1D 或 2D 数组")

        # --------- 画布尺寸（是否包含 colorbar）---------
        two_panels = heat is not None
        if two_panels:
            extra = (cbar_pad + cbar_w) if include_cbar else 0
            TOT_W, TOT_H = W_cell * 2 + gap + extra, H_cell
        else:
            TOT_W, TOT_H = W_cell, H_cell

        # --------- 缓存不同布局 ---------
        cache_attr = "_mpl_cache_dual_cbar" if two_panels else "_mpl_cache_single"
        cache = getattr(self, cache_attr, None)
        if cache is None or cache.get("TOT_W") != TOT_W or cache.get("TOT_H") != TOT_H or cache.get("include_cbar") != include_cbar:
            dpi = 100
            fig = Figure(figsize=(TOT_W / dpi, TOT_H / dpi), dpi=dpi)
            canvas = FigureCanvas(fig)
            axes = {}

            if two_panels:
                # 左轴
                axes["L"] = fig.add_axes([0, 0, W_cell / TOT_W, 1])
                # 右轴
                axes["R"] = fig.add_axes([(W_cell + gap) / TOT_W, 0, W_cell / TOT_W, 1])
                if include_cbar:
                    # colorbar 轴（竖向）
                    cbar_x0 = (W_cell + gap + W_cell + cbar_pad) / TOT_W
                    axes["C"] = fig.add_axes([cbar_x0, 0.1, cbar_w / TOT_W, 0.8])  # 留上下边距好看些
                for ax in axes.values():
                    ax.set_xlim(0, W_cell)
                    ax.set_ylim(H_cell, 0)
                    ax.axis("off")
            else:
                axes["L"] = fig.add_axes([0, 0, 1, 1])
                axes["L"].set_xlim(0, W_cell)
                axes["L"].set_ylim(H_cell, 0)
                axes["L"].axis("off")

            cache = {"fig": fig, "canvas": canvas, "axes": axes,
                    "TOT_W": TOT_W, "TOT_H": TOT_H, "include_cbar": include_cbar}
            setattr(self, cache_attr, cache)

        fig, canvas, axes = cache["fig"], cache["canvas"], cache["axes"]
        axL = axes["L"]
        axL.clear(); axL.set_xlim(0, W_cell); axL.set_ylim(H_cell, 0); axL.axis("off")

        # --------- 左侧棋盘（数字更小）---------
        axL.add_patch(Rectangle((0, 0), W_cell, H_cell,
                                facecolor=_rgb255_to_1(self.colors["bg"]), edgecolor=None))
        for y in range(self.height):
            for x in range(self.width):
                x0, y0 = x * self.cell_size, y * self.cell_size
                v = int(self.board[x, y])

                if v == 10:
                    axL.add_patch(Rectangle((x0, y0), self.cell_size, self.cell_size,
                                            facecolor=_rgb255_to_1(self.colors["unknown"]), edgecolor=None))
                elif v == 9:
                    cx, cy = x0 + self.cell_size / 2, y0 + self.cell_size / 2
                    r = max(2, self.cell_size // 3)
                    axL.add_patch(Circle((cx, cy), r, color=_rgb255_to_1(self.colors["mine"])))
                else:
                    if v != 0:
                        cx, cy = x0 + self.cell_size / 2, y0 + self.cell_size / 2
                        col = _rgb255_to_1(self.colors["numbers"][v])
                        axL.text(cx, cy, str(v), ha="center", va="center",
                                color=col,
                                fontsize=max(6, self.cell_size * float(number_text_scale)),
                                family="DejaVu Sans", weight="bold")

                axL.add_patch(Rectangle((x0, y0), self.cell_size, self.cell_size,
                                        fill=False, linewidth=1, edgecolor=_rgb255_to_1(self.colors["grid"])))

        # --------- 右侧热力图 + colorbar ---------
        if two_panels:
            axR = axes["R"]
            axR.clear(); axR.set_xlim(0, W_cell); axR.set_ylim(H_cell, 0); axR.axis("off")

            vmin = float(np.nanmin(heat)) if np.isfinite(heat).any() else 0.0
            vmax = float(np.nanmax(heat)) if np.isfinite(heat).any() else 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-6

            im = axR.imshow(
                heat,
                origin="upper",
                extent=(0, W_cell, H_cell, 0),
                cmap="viridis",
                vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )

            # 网格
            for y in range(self.height):
                for x in range(self.width):
                    x0, y0 = x * self.cell_size, y * self.cell_size
                    axR.add_patch(Rectangle((x0, y0), self.cell_size, self.cell_size,
                                            fill=False, linewidth=1, edgecolor=_rgb255_to_1(self.colors["grid"])))

            # 概率文字（更小）
            if show_prob_text:
                mid = 0.5 * (vmin + vmax)
                fs_prob = max(6, int(self.cell_size * float(prob_text_scale)))
                for y in range(self.height):
                    for x in range(self.width):
                        val = float(heat[y, x])
                        cx, cy = x * self.cell_size + self.cell_size / 2, y * self.cell_size + self.cell_size / 2
                        txt_color = (1, 1, 1) if val >= mid else (0, 0, 0)
                        axR.text(cx, cy, f"{val:.2f}", ha="center", va="center",
                                color=txt_color, fontsize=fs_prob, family="DejaVu Sans")

            # 高亮 action
            ax_idx = None
            if action is not None:
                ax_idx = int(action)
            else:
                if np.isfinite(heat).any():
                    flat_idx = int(np.nanargmax(heat))
                    y_sel = flat_idx // self.width
                    x_sel = flat_idx % self.width
                    ax_idx = x_sel * self.height + y_sel
            if ax_idx is not None:
                x_sel = ax_idx // self.height
                y_sel = ax_idx % self.height
                for _ax in (axL, axR):
                    x0, y0 = x_sel * self.cell_size, y_sel * self.cell_size
                    _ax.add_patch(Rectangle((x0, y0), self.cell_size, self.cell_size,
                                            fill=False, linewidth=2.5, edgecolor=(1.0, 1.0, 0.0)))

            # colorbar
            if include_cbar and "C" in axes:
                cax = axes["C"]
                cax.clear()
                cb = fig.colorbar(im, cax=cax, orientation="vertical")
                cb.formatter = mticker.FormatStrFormatter('%.2f')
                cb.update_ticks()
                cax.yaxis.set_ticks_position('right')  # 刻度在右侧
                cax.tick_params(labelsize=max(6, int(self.cell_size * 0.30)))

        # --------- 输出为 RGB numpy ---------
        canvas.draw()
        w, h = canvas.get_width_height()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = rgba[..., :3].copy()

        if mode == "print":
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    v = int(self.board[x, y])
                    row.append('.' if v == 10 else ('*' if v == 9 else str(v)))
                print(" ".join(row))
            return getattr(self, "board", None).copy()

        if mode == "human":
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(frame)
            plt.axis("off")
            plt.show(block=False)
            return frame

        return frame  # 'rgb_array'

        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            
    def get_action_mask(self):
        return self.action_mask.copy()
    
    def _update_mask_dfs(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                    continue

                if (new_x, new_y) in self.mines:
                    continue
                action = new_x * self.height + new_y
                if not self.action_mask[action]:
                    continue  # 已经揭开过

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
    
    def _build_true_board(self):
        b = np.zeros((self.width, self.height), dtype=int)
        for x in range(self.width):
            for y in range(self.height):
                b[x, y] = 9 if (x, y) in self.mines else self.count_mines(x, y)
        return b
