from env.minesweeper import MinesweeperEnv
from src import *

def main(solver_type='base'):
    # 创建环境
    if solver_type != 'logic':
        env = MinesweeperEnv(width=8, height=8, num_mines=10, use_dfs=True)
    else:
        env = MinesweeperEnv(width=8, height=8, num_mines=10, use_dfs=False)
    
    if solver_type == 'base':
        solver = MinesweeperSolver(env)
    elif solver_type == 'logic':
        solver = LogicSolver(env)
    
    # 运行一个回合
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 获取动作
        if solver_type == 'logic':
            action = solver.get_action(env)
        else:
            action = solver.get_action(state)
        if action is None:
            break
            
        # 执行动作
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # 显示当前状态
        # print("\nBoard state:")
        env.render(mode='pygame')
        # print(f"Reward: {reward}")
        
    print(f"\nGame Over! Total reward: {total_reward}")

if __name__ == "__main__":
    main(solver_type='logic')