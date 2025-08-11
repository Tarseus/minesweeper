from typing import Optional, Dict, Any
import os, time, random
import numpy as np
import torch
import gymnasium as gym

from src.utils.env_utils import make_env
from src.wrappers.video_record import VideoRecorderWrapper
from src.models import CNNBased
from src.algo.ppo import PPO
from src.config import PPOConfig

def test():
    import random
    config = PPOConfig()
    config.track = False
    seed = random.randint(0, 2**32 - 1)
    # seed = config.seed
    run_name = f"{config.exp_name}_{config.seed}_{time.strftime('%d/%m/%Y_%H-%M-%S')}"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda:5" if config.cuda and torch.cuda.is_available() else "cpu")

    if run_name is None:
        run_name = f"{config.exp_name}_{seed}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"

    envs = gym.vector.SyncVectorEnv([
        make_env(config, seed + i, i, False, run_name) for i in range(1000)
    ])

    H, W = envs.single_observation_space.shape
    model = CNNBased(obs_shape=(H, W)).to(device)
    model = torch.compile(model, mode="max-autotune").eval()

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

    # if getattr(config, "use_pretrain", False):
    agent.load(config.test_model_path)

    out = agent.evaluate_n_episodes()
    agent.evaluate_video()

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
    test()