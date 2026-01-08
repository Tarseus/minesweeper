from typing import Optional, Dict, Any
import os, time, random
import numpy as np
import torch
import gymnasium as gym
import argparse

from src.utils.env_utils import make_env
from src.wrappers.video_record import VideoRecorderWrapper
from src.models import CNNBased, TransformerBasedModel, GridGNNBased
from src.algo.ppo import PPO
from src.algo.ce import CE
from src.config import PPOConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO/CE on specified GPU.")
    parser.add_argument(
        "--gpu", type=int, default=0, help="Specify the GPU to train on (default: 2)."
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="beginner",
        choices=["beginner", "intermediate", "expert", "curriculum"],
        help="Game difficulty level, or curriculum (beginner->intermediate->expert).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "ce"],
        help="Training algorithm: ppo (RL) or ce (cross-entropy).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["transformer", "cnn", "gnn"],
        help="Backbone model: transformer/cnn/gnn.",
    )
    return parser.parse_args()

def train(gpu: int, difficulty: str, algo: str = "ppo", model_name: str = "transformer"):
    algo_lower = algo.lower()
    model_name = (model_name or "transformer").lower()

    if difficulty == "curriculum" and algo_lower != "ppo":
        raise ValueError("difficulty=curriculum currently supports --algo ppo only.")
    if difficulty == "curriculum" and model_name == "transformer":
        raise ValueError("difficulty=curriculum requires --model gnn or --model cnn (transformer is shape-fixed).")

    stage_configs = None
    if difficulty == "curriculum":
        stage_difficulties = ["beginner", "intermediate", "expert"]
        stage_configs = [PPOConfig(difficulty=d) for d in stage_difficulties]
        config = stage_configs[0]
        config.exp_name = "ms_ai_ppo_curriculum"
        run_name = f"{config.exp_name}_{config.seed}_{time.strftime('%d/%m/%Y_%H-%M-%S')}"
        seed = config.seed + sum(c.total_timesteps for c in stage_configs)
    else:
        config = PPOConfig(difficulty=difficulty)
        if algo_lower == "ce":
            config.exp_name = "ms_ai_ce_" + config.difficulty
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

    def build_envs(cfg, base_seed: int):
        return gym.vector.SyncVectorEnv(
            [make_env(cfg, base_seed + i, i, False, run_name) for i in range(cfg.num_envs)]
        )

    envs = build_envs(config, seed)

    H, W = envs.single_observation_space.shape
    if model_name == "transformer":
        model = TransformerBasedModel(obs_shape=(H, W)).to(device)
        model = torch.compile(model, mode="max-autotune")
    elif model_name == "cnn":
        model = CNNBased(obs_shape=(H, W)).to(device)
        model = torch.compile(model, mode="max-autotune")
    elif model_name == "gnn":
        gnn_obs_shape = None if difficulty == "curriculum" else (H, W)
        model = GridGNNBased(obs_shape=gnn_obs_shape).to(device)
        model = torch.compile(model, mode="max-autotune")
    else:
        raise ValueError(f"Unknown model: {model_name}")

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

    if algo_lower == "ppo":
        AgentCls = PPO
    elif algo_lower == "ce":
        AgentCls = CE
    else:
        raise ValueError(f"Unknown algo: {algo}")

    agent = AgentCls(
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

    if difficulty != "curriculum":
        agent.train()
    else:
        for stage_idx, stage_cfg in enumerate(stage_configs):
            if stage_idx > 0:
                old_envs = envs
                envs = build_envs(stage_cfg, seed + 100_000 * stage_idx)
                old_envs.close()
                agent.set_envs(envs, stage_cfg)
            print(
                f"[curriculum] stage={stage_cfg.difficulty} "
                f"size={stage_cfg.width}x{stage_cfg.height} mines={stage_cfg.num_mines}"
            )
            agent.train()

    if wandb_run is not None:
        tag = "curriculum" if difficulty == "curriculum" else config.difficulty
        final_path = os.path.join(wandb_run.dir, f"ppo_{tag}_{agent.state.global_step}.pth")
    else:
        os.makedirs("checkpoints", exist_ok=True)
        tag = "curriculum" if difficulty == "curriculum" else config.difficulty
        final_path = os.path.join("checkpoints", f"ppo_{tag}_{agent.state.global_step}.pth")
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
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    train(args.gpu, args.difficulty, args.algo, args.model)
