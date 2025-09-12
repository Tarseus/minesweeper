from typing import Optional, Dict, Any
import os, time, random
import numpy as np
import torch
import gymnasium as gym
import argparse

from src.utils.env_utils import make_env
from src.wrappers.video_record import VideoRecorderWrapper
from src.models import CNNBased, TransformerBasedModel
from src.algo.ppo import PPO
from src.config import PPOConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on specified GPU.")
    parser.add_argument(
        "--gpu", type=int, default=0, help="Specify the GPU to train on (default: 2)."
    )
    parser.add_argument(
        "--difficulty", type=str, default="beginner", help="Game difficulty level (default: beginner)."
    )
    return parser.parse_args()

def train(gpu: int, difficulty: str):
    config = PPOConfig(difficulty=difficulty)
    run_name = f"{config.exp_name}_{config.seed}_{time.strftime('%d/%m/%Y_%H-%M-%S')}"
    seed = config.seed + config.total_timesteps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # 使用从命令行获得的 GPU
    device = torch.device(f"cuda:{gpu}" if config.cuda and torch.cuda.is_available() else "cpu")

    if run_name is None:
        run_name = f"{config.exp_name}_{seed}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"

    envs = gym.vector.SyncVectorEnv([ 
        make_env(config, seed + i, i, False, run_name) for i in range(config.num_envs)
    ])

    H, W = envs.single_observation_space.shape
    model = TransformerBasedModel(obs_shape=(H, W)).to(device)
    model = torch.compile(model, mode="max-autotune")

    writer = None
    wandb_run = None
    if getattr(config, "track", False):
        from torch.utils.tensorboard import SummaryWriter
        import wandb
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value||-|-|" + "".join([f"|{k}|{v}|" for k, v in vars(config).items()]),
        )
        wandb_run = wandb.init(
            project=config.wandb_project,
            sync_tensorboard=True,
            config=vars(config),
            name=config.exp_name,
            monitor_gym=True,
            save_code=True,
            mode=os.environ.get("WANDB_MODE", "offline"),
        )

    val_env = None
    if getattr(config, "capture_video", False):
        from src.env import MinesweeperEnv
        val_env = MinesweeperEnv(config)
        val_env = VideoRecorderWrapper(
            val_env,
            videos_dir=f"videos/{run_name}",
            fps=1,
            name_prefix=f"val_0",
            if_save_frames=True,
        )

    agent = PPO(
        envs=envs,
        model=model,
        config=config,
        device=device,
        writer=writer,
        wandb_run=wandb_run,
        val_env=val_env,
        video_wrapper_cls=VideoRecorderWrapper,
        run_name=run_name,
    )

    if getattr(config, "use_pretrain", False):
        pre_path = config.pretrain_model_path 
        print(f"Loading pre-trained model from {config.pretrain_model_path}")
        agent.load(pre_path)

    agent.train()

    if wandb_run is not None:
        final_path = os.path.join(wandb_run.dir, f"ppo_{config.difficulty}_{agent.state.global_step}.pth")
    else:
        os.makedirs("checkpoints", exist_ok=True)
        final_path = os.path.join("checkpoints", f"ppo_{config.difficulty}_{agent.state.global_step}.pth")
    agent.save(final_path)

    envs.close()
    if writer is not None:
        writer.close()

    out = {
        "final_path": final_path,
        "run_name": run_name,
        "global_step": agent.state.global_step,
        "win_rate": agent.state.win_rate,
    }
    return out

if __name__ == "__main__":
    args = parse_args()
    train(args.gpu, args.difficulty)
