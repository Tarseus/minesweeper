import numpy as np

class MinesweeperSolver:
    def __init__(self, env):
        self.env = env
        self.width = env.width
        self.height = env.height
        self.num_mines = env.num_mines
        
    def get_action(self, state):
        """
        基于当前状态选择下一步动作
        Args:
            state: 游戏当前状态 (numpy array)
        Returns:
            action: 选择的动作 (int)
        """
        # 获取当前可用的动作
        valid_actions = np.where(self.env.action_mask)[0]
        
        if len(valid_actions) == 0:
            return None
            
        if np.all(state == 10):
            center_x = self.width // 2
            center_y = self.height // 2
            print(f"Choose center: ({center_x}, {center_y})")
            return center_x * self.height + center_y
        # 简单策略：随机选择一个有效动作
        return np.random.choice(valid_actions)