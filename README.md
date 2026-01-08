# Minesweeper RL

## Quick start

### Train (PPO / CE)

- PPO + GNN（尺寸无关、局部消息传递推理）
  - `python train.py --gpu 0 --difficulty beginner --algo ppo --model gnn`
- PPO + GNN 课程学习（beginner->intermediate->expert）
  - `python train.py --gpu 0 --difficulty curriculum --algo ppo --model gnn`
- PPO + Transformer（默认）
  - `python train.py --gpu 0 --difficulty beginner --algo ppo --model transformer`
- CE（交叉熵蒸馏/行为克隆式目标，见 `src/algo/ce.py`）
  - `python train.py --gpu 0 --difficulty beginner --algo ce --model gnn`

训练日志默认写入 `runs/`（如开启 wandb 也会写入 wandb 目录），模型会保存到 `checkpoints/`。
