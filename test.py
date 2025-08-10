from typing import Optional, Dict, Any, Tuple
import os, time, random, glob, re
import numpy as np
import torch
import gymnasium as gym

from src.utils.env_utils import make_env
from src.wrappers.video_record import VideoRecorderWrapper
from src.models import CNNBased
from src.algo.ppo import PPO
from src.config import PPOConfig


def _find_latest_checkpoint(pattern: str) -> Optional[str]:
    """在 checkpoints/ 里按修改时间寻找最新权重."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _safe_load_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """尽可能鲁棒地加载各种保存格式的 checkpoint 到 model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    loaded = False

    # 1) 直接当成 state_dict
    if isinstance(ckpt, dict):
        # 常见几种命名
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                try:
                    model.load_state_dict(ckpt[key], strict=False)
                    loaded = True
                    break
                except Exception:
                    pass
        if not loaded:
            # 可能就是 state_dict 本体
            try:
                model.load_state_dict(ckpt, strict=False)
                loaded = True
            except Exception:
                pass
    if not loaded:
        raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")

def test() -> Dict[str, Any]:
    """
    基于 train() 的评估脚本。
    - model_path: 指定待评估的权重路径；若为 None，则会在 checkpoints/ 中自动搜寻当前难度下最新的 ppo_*.pth
    - num_episodes: 评估回合数
    返回：包含平均回报、胜率、视频目录等信息的字典
    """
    num_episodes = 1
    config = PPOConfig()
    run_name = f"{config.exp_name}_test_{config.seed}_{time.strftime('%d/%m/%Y_%H-%M-%S')}"
    seed = config.seed

    # 随机种子与设备
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    device = torch.device("cuda:2" if getattr(config, "cuda", False) and torch.cuda.is_available() else "cpu")

    model_path = getattr(config, "test_model_path", None)
    if model_path is None or not os.path.isfile(model_path):
        os.makedirs("checkpoints", exist_ok=True)
        pattern = os.path.join("checkpoints", f"ppo_{getattr(config, 'difficulty', '*')}_*.pth")
        model_path = _find_latest_checkpoint(pattern)

    if model_path is None or not os.path.isfile(model_path):
        raise FileNotFoundError(
            "未找到可用的模型权重。请传入 model_path 或在 checkpoints/ 下放置 ppo_*.pth 文件。"
        )

    # 先构造一个临时环境获取观测空间形状
    base_env_fn = make_env(config, seed, 0, False, run_name)
    tmp_env = base_env_fn()
    try:
        obs0, _ = tmp_env.reset()
        H, W = tmp_env.observation_space.shape[:2]
    finally:
        tmp_env.close()

    # 构建模型并加载权重（保持与 train 一致）
    model = CNNBased(obs_shape=(H, W)).to(device)
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        # torch < 2.0 或环境不支持 compile 时，安全降级
        pass
    _safe_load_model(model, model_path, device)
    model.eval()

    # 若需要使用 PPO 容器（例如你的项目里 act/predict 在 PPO 里），在此构建一个“空训练”的 agent 仅用于推理
    # 注意：这里不创建 vector env，也不调用 train()。
    agent = None
    try:
        agent = PPO(
            envs=None,  # 有的实现允许为 None；若不允许，可以删除 agent 相关代码，仅依赖 model 前向
            model=model,
            config=config,
            device=device,
            writer=None,
            wandb_run=None,
            val_env=None,
            video_wrapper_cls=VideoRecorderWrapper,
            run_name=run_name,
        )
        # 尝试用 agent 自带的加载（不强制，有些工程需要）
        try:
            agent.load(model_path)
        except Exception:
            pass
    except Exception:
        # 如果 PPO 构造需要 envs 等，这里兜底为 None，不影响后续仅用 model 推理
        agent = None

    # 评估循环
    rewards, lengths, wins = [], [], []
    video_dir = f"videos/{run_name}" if getattr(config, "capture_video", False) else None
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    for ep in range(num_episodes):
        # 每回合单独创建环境；如需录屏则包一层 VideoRecorderWrapper
        env = base_env_fn()
        if video_dir:
            env = VideoRecorderWrapper(
                env,
                videos_dir=video_dir,
                fps=1,
                name_prefix=f"test_{ep}",
                if_save_frames=True,
            )

        obs, info = env.reset()
        done = False
        ep_ret, ep_len, ep_win = 0.0, 0, 0

        while not done:
            obs = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            action, probs, entropy = model.get_action(obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action, probs=probs.probs.squeeze(0))
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1

            # 胜负信息：兼容多种键
            if isinstance(info, dict):
                for k in ("win", "is_success", "victory", "won"):
                    if k in info:
                        ep_win = int(bool(info[k]))
                        break

        rewards.append(ep_ret)
        lengths.append(ep_len)
        wins.append(ep_win)
        env.close()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    win_rate = float(np.mean(wins)) if wins else 0.0

    out = {
        "model_path": model_path,
        "run_name": run_name,
        "episodes": num_episodes,
        "mean_reward": mean_reward,
        "mean_length": mean_len,
        "win_rate": win_rate,
        "video_dir": video_dir,
    }
    print(
        f"[TEST] episodes={num_episodes} | mean_reward={mean_reward:.3f} | mean_len={mean_len:.1f} | win_rate={win_rate:.3f}"
    )
    return out


if __name__ == "__main__":
    # 示例：指定权重或自动在 checkpoints/ 下寻找最新
    test()  # 或 test(model_path="checkpoints/ppo_easy_100000.pth", num_episodes=20)
