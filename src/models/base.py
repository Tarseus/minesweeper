import os, sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)
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
        
        if np.all(state[0] == 10):
            center_x = self.width // 2
            center_y = self.height // 2
            print(f"Choose center: ({center_x}, {center_y})")
            return center_x * self.height + center_y
        
        return np.random.choice(valid_actions)
    
# if __name__ == "__main__":
#     from env.minesweeper import MinesweeperEnv

#     env = MinesweeperEnv(
#         width=30, 
#         height=16, 
#         num_mines=99, 
#         use_dfs=True
#     )
#     solver = MinesweeperSolver(env)
    
#     # 运行一个回合
#     state = env.reset()
#     done = False
#     total_reward = 0
    
#     while not done:
#         action = solver.get_action(state)
#         if action is None:
#             break
            
#         state, reward, done, _ = env.step(action)
#         total_reward += reward
        
#         env.render(mode='pygame')
        
#     print(f"\nGame Over! Total reward: {total_reward}")