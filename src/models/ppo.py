import sys
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["WANDB_MODE"] = "offline"
# Add the src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from wrappers.video_record import VideoRecorderWrapper
from config.ppo_config import PPOConfig
import numpy as np
import gymnasium as gym
import torch
import random
from env.minesweeper import MinesweeperEnv
import torch.nn as nn
from torch.distributions import Categorical
from utils.env_utils import make_env
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from models.logic import LogicSolver

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

class SqueezeExcite(nn.Module):
    """通道注意力 (SE)"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SqueezeExcite(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return self.relu(x + y)


class Agent(nn.Module):
    """
    9×9–10 雷（或任意尺寸）扫雷智能体：
      - 输入：one-hot 11 通道 (B,11,H,W)
      - Policy：1×1 Conv → (B,1,H,W) → 展平成 (B,H*W) logits
      - Value：GAP → 64 → FC → 1
    """
    def __init__(self, envs):
        super().__init__()
        h, w = envs.single_observation_space.shape         # board size
        self.board_size = h * w
        in_ch = 11                                         # one-hot 通道数

        # ─── Backbone ──────────────────────────────────────────────────────────
        layers = [
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        # 6 个 ResBlock 足够；若想更深可自行加
        for _ in range(7):
            layers.append(ResBlock(64))
        self.backbone = nn.Sequential(*layers)

        # ─── Heads ─────────────────────────────────────────────────────────────
        self.policy_head = nn.Conv2d(64, 1, kernel_size=1)       # (B,1,H,W)

        self.value_head = nn.Sequential(                         # (B,64,1,1)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                                        # → (B,64)
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    # --------------------------------------------------------------------- #
    # ↓↓↓  Utility  ↓↓↓
    # --------------------------------------------------------------------- #
    @staticmethod
    def _one_hot(obs: torch.Tensor) -> torch.Tensor:
        """把 (B,H,W) board int 转 one-hot → (B,11,H,W)"""
        return F.one_hot(obs.long(), num_classes=11).permute(0, 3, 1, 2).float()

    # --------------------------------------------------------------------- #
    # ↓↓↓  Public API  ↓↓↓
    # --------------------------------------------------------------------- #
    def get_value(self, x):
        feat = self.backbone(self._one_hot(x))
        return self.value_head(feat)

    def get_action_and_value(self, x, action=None, action_mask=None):
        """
        Args:
            x            : (B,H,W) board
            action       : optional pre-selected tensor
            action_mask  : list/ndarray[(H*W,)]  True=legal / False=illegal
        Returns:
            action, log_prob, entropy, value
        """
        B, H, W = x.shape
        device  = x.device

        # “首步点中心”启发式
        if torch.all(x == 10):
            center = (H // 2) * W + (W // 2)
            action = torch.full((B,), center, dtype=torch.long, device=device)

        feat   = self.backbone(self._one_hot(x))
        logits = self.policy_head(feat).flatten(1)          # (B,H*W)

        # 掩掉非法动作
        if action_mask is not None:
            mask = torch.as_tensor(np.vstack(action_mask), dtype=torch.bool, device=device)
            logits = logits.masked_fill(~mask, -1e9)

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        elif not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.long, device=device)

        value = self.value_head(feat)
        return action, probs.log_prob(action), probs.entropy(), value
    
if __name__ == "__main__":
    config = PPOConfig()
    run_name = f"{config.exp_name}_{config.seed}_{time.strftime('%d/%m/%Y_%H-%M-%S')}"
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project, 
            sync_tensorboard=True,
            config=vars(config),
            name=config.exp_name,
            monitor_gym=True,
            save_code=True,
            mode="offline",
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(config, config.seed + i, i, False, run_name) for i in range(config.num_envs)]
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only Discrete action spaces are supported"

    agent = Agent(envs).to(device)
    if config.use_pretrain:
        agent.load_state_dict(torch.load(config.pretrain_model_path + "_" + config.difficulty + ".pth"))
        # agent.load_state_dict(torch.load(config.pretrain_model_path + ".pth"))
    if config.track:
        wandb.watch(agent, log="all", log_freq=1000)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    
    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    
    obs_val = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions_val = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_val = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values_val = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards_val = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones_val = torch.zeros((config.num_steps, config.num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    
    next_done = torch.zeros(config.num_envs).to(device)
    num_updates = config.total_timesteps // config.batch_size
    # progress_bar = tqdm(range(num_updates), dynamic_ncols=True)

    val_env = MinesweeperEnv(config)
    val_env = VideoRecorderWrapper(val_env, 
                                    videos_dir = f"videos/{run_name}",
                                    fps=1,
                                    name_prefix=f"val_{global_step}",
                                    if_save_frames=True,
                                )
    for update in range(1, num_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = config.learning_rate * frac
            optimizer.param_groups[0]['lr'] = lrnow
        
        # rollout trajectory
        obs1, info = envs.reset()
        action_masks = info["action_mask"]
        next_obs = torch.Tensor(obs1).to(device)
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, 
                                                                       action_mask=action_masks)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            action_masks = info["action_mask"]
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in info:
                total_count = 0
                total_win = 0
                for item in info["final_info"]:
                    if item is not None:
                        total_count += 1
                        if item.get("is_success", False):
                            total_win += 1
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                win_rate = total_win / total_count
                writer.add_scalar("charts/win_rate", win_rate, global_step)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if config.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + config.gamma * nextnonterminal * next_return
                advantages = returns - values
            
            # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.update_epoches):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.mini_batch_size):
                end = start + config.mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_value_loss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # progress_bar.set_postfix({
        #     'value_loss': f"{v_loss.item():.3f}",
        #     'policy_loss': f"{pg_loss.item():.3f}",
        #     'entropy': f"{entropy_loss.item():.3f}",
        #     'var': f"{explained_var:.3f}",
        #     'SPS': f"{int(global_step / (time.time() - start_time))}",
        #     'win_rate': f"{win_rate:.3f}",
        # })
        elapsed = time.time() - start_time
        progress = update / num_updates
        if progress > 0:
            total_estimated = elapsed / progress
            remaining = total_estimated - elapsed
        else:
            remaining = 0
        print(f"Update {update}/{num_updates} | "
            f"Value Loss: {v_loss.item():.3f} | "
            f"Policy Loss: {pg_loss.item():.3f} | "
            f"Entropy: {entropy_loss.item():.3f} | "
            f"Var: {explained_var:.3f} | "
            f"SPS: {int(global_step / (time.time() - start_time))} | "
            f"Win Rate: {win_rate:.3f} | "
            f"ETA: {format_seconds(remaining)}")
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if update % config.save_freq == 0 and config.track:
            model_path = os.path.join(wandb.run.dir, f"ppo_{config.difficulty}_{global_step}.pth")
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        if update % config.capture_video_freq == 0:
            if config.capture_video:
                obs2, info_val = val_env.reset(name_prefix=f"val_{global_step}")
                next_obs_val = torch.Tensor(obs2).unsqueeze(0).to(device)
                next_done_val = torch.zeros(1).to(device)
                action_masks_val = [info_val["action_mask"]]

                while True:
                    with torch.no_grad():
                        action_val, logprob_val, _, value_val = agent.get_action_and_value(next_obs_val, 
                                                                                action_mask=action_masks_val)
                        
                    next_obs_val, reward_val, terminated, truncated, info_val = val_env.step(action_val[0].cpu().numpy())
                    action_masks_val = [info_val["action_mask"]]
                    done_val = np.logical_or(terminated, truncated)
                    
                    if done_val:
                        break
                    next_obs_val = torch.Tensor(next_obs_val).unsqueeze(0).to(device)
                    next_done_val = torch.Tensor([done_val]).to(device)
                
                time.sleep(1)
                val_env.close()
        
    envs.close()
    writer.close()