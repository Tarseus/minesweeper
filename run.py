from env.minesweeper import MinesweeperEnv
from src import MinesweeperSolver

def main():
    # 创建环境
    env = MinesweeperEnv(width=8, height=8, num_mines=10)
    # 创建求解器
    solver = MinesweeperSolver(env)
    
    # 运行一个回合
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 获取动作
        action = solver.get_action(state)
        if action is None:
            break
            
        # 执行动作
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # 显示当前状态
        print("\nBoard state:")
        env.render(mode='pygame')
        print(f"Reward: {reward}")
        
    print(f"\nGame Over! Total reward: {total_reward}")

if __name__ == "__main__":
    main()