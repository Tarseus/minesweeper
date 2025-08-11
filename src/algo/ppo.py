from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# local imports
# from models.minesweeper_model import MinesweeperModel


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class TrainState:
    global_step: int = 0
    win_rate: float = 0.0
    episodic_return: float = 0.0
    episodic_length: float = 0.0


class PPO:
    """
    Encapsulates PPO training/evaluation for MinesweeperModel.
    Keeps your original hyperparameter semantics (names) from PPOConfig.
    """
    def __init__(
        self,
        envs,
        model: nn.Module,
        config,
        device: torch.device,
        writer: Optional[SummaryWriter] = None,
        wandb_run: Optional[object] = None,
        val_env=None,
        video_wrapper_cls=None,
        run_name: Optional[str] = None,
    ):
        self.envs = envs
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.writer = writer
        self.wandb_run = wandb_run
        self.val_env = val_env
        self.video_wrapper_cls = video_wrapper_cls
        self.run_name = run_name or "run"

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        
        self.phase = config.phase  # "train" or "test"

        # rollout buffers
        shape_obs = envs.single_observation_space.shape
        self.obs      = torch.zeros((config.num_steps, config.num_envs) + shape_obs, device=device)
        self.actions  = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape, device=device)
        self.logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.values   = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.rewards  = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.dones    = torch.zeros((config.num_steps, config.num_envs), device=device)

        self.state = TrainState(global_step=0, win_rate=0.0)

    # ------------------------ core methods ------------------------
    def rollout(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.config
        device = self.device

        next_done = torch.zeros(cfg.num_envs, device=device)
        obs_np, info = self.envs.reset()
        B = obs_np.shape[0]
        A = self.envs.single_action_space.n
        action_masks = self._norm_mask(info["action_mask"], B, A, self.device)
        next_obs = torch.as_tensor(obs_np, device=device)

        final_info_seen = 0
        total_win = 0
        total_return = 0.0
        total_length = 0

        for step in range(cfg.num_steps):
            self.state.global_step += cfg.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.model.get_action_and_value(
                    next_obs, action_mask=action_masks
                )
                self.values[step] = value.flatten()

            self.actions[step] = action
            self.logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            action_masks = self._norm_mask(info["action_mask"], B, A, self.device)
            done = np.logical_or(terminated, truncated)

            self.rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, device=device)
            next_done = torch.as_tensor(done, device=device, dtype=self.rewards.dtype)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item is not None:
                        final_info_seen += 1
                        if item.get("is_success", False):
                            total_win += 1
                        if item.get("episode") is not None:
                            total_return += item["episode"]["r"]
                            total_length += item["episode"]["l"]
                            
        win_rate = (total_win / final_info_seen) if final_info_seen > 0 else self.state.win_rate
        episodic_return = total_return / final_info_seen if final_info_seen > 0 else 0.0
        episodic_length = total_length / final_info_seen if final_info_seen > 0 else 0.0
        self.state.win_rate = win_rate
        self.state.episodic_return = episodic_return
        self.state.episodic_length = episodic_length

        with torch.no_grad():
            next_value = self.model.get_value(next_obs).reshape(1, -1)

        out = {
            "next_obs": next_obs,
            "next_done": next_done,
            "next_value": next_value,
        }
        return out["next_obs"], out

    def compute_advantages(self, next_done, next_value):
        cfg = self.config
        rewards, values, dones = self.rewards, self.values, self.dones
        next_done = next_done.to(dtype=rewards.dtype)
        dones = dones.to(dtype=rewards.dtype)

        if cfg.gae:
            advantages = torch.zeros_like(rewards, device=self.device)
            lastgaelam = 0.0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards, device=self.device)
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + cfg.gamma * nextnonterminal * next_return
            advantages = returns - values
        return advantages, returns

    def update_policy(self, advantages, returns):
        cfg = self.config

        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        old_approx_kl = torch.tensor(0.0)

        for epoch in range(cfg.update_epoches):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_value_loss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = self.values.reshape(-1).detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        stats = {
            "v_loss": float(v_loss.item()),
            "pg_loss": float(pg_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "old_approx_kl": float(old_approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs) if clipfracs else 0.0),
            "explained_var": float(explained_var),
        }
        return stats

    def maybe_eval_and_save(self, update_idx: int):
        cfg = self.config
        # save
        if (update_idx % cfg.save_freq == 0) and getattr(cfg, "track", False) and self.wandb_run is not None:
            model_path = os.path.join(self.wandb_run.dir, f"ppo_{cfg.difficulty}_{self.state.global_step}.pth")
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        # video eval
        if (update_idx % cfg.capture_video_freq == 0) and getattr(cfg, "capture_video", False) and self.val_env is not None:
            self.evaluate_video(prefix=f"val_{self.state.global_step}")

    @torch.no_grad()
    def evaluate_n_episodes(self, total_episodes: int = 100_000, prefix: str = "test"):
        """
        用现有的 self.envs（向量环境，自动 reset done 的那种）评测 total_episodes 局。
        返回：
            {
            "win_rate": float,
            "wins": np.ndarray[bool, total_episodes],
            "returns": np.ndarray[float32, total_episodes],
            "lengths": np.ndarray[int32, total_episodes],
            }
        """
        device = self.device
        envs = self.envs

        # ---- reset &基本信息 ----
        obs_np, info = envs.reset()
        B = obs_np.shape[0]
        A = envs.single_action_space.n
        obs = torch.as_tensor(obs_np, device=device)
        action_masks = self._norm_mask(info["action_mask"], B, A, device)

        # ---- 结果缓冲区（若想省内存可用 np.memmap 落盘）----
        wins    = np.zeros(total_episodes, dtype=bool)
        returns = np.zeros(total_episodes, dtype=np.float32)
        lengths = np.zeros(total_episodes, dtype=np.int32)
        # 示例：用 memmap
        # wins    = np.memmap("wins.bool",    mode="w+", dtype=bool,     shape=(total_episodes,))
        # returns = np.memmap("returns.f32",  mode="w+", dtype=np.float32, shape=(total_episodes,))
        # lengths = np.memmap("lengths.i32",  mode="w+", dtype=np.int32,   shape=(total_episodes,))

        # 若环境不提供 episode 统计，这两条作为兜底在线累计
        run_ret = np.zeros(B, dtype=np.float32)
        run_len = np.zeros(B, dtype=np.int32)

        finished = 0  # 已完成的局数
        ptr = 0       # 写入结果数组的游标

        def _get_success(fi: dict) -> bool:
            # 兼容多种 key
            for k in ("is_success", "success", "won", "win"):
                if k in fi:
                    return bool(fi[k])
            return False

        while finished < total_episodes:
            # 前向选择动作（贪心/你模型默认策略）
            action, logprob, _, value = self.model.get_action_and_value(
                obs, action_mask=action_masks
            )

            # 环境前进一步（大多数向量 env 会对 done 的实例自动 reset）
            next_obs_np, reward, terminated, truncated, info = envs.step(action.detach().cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # 兜底的在线累计
            r = np.asarray(reward, dtype=np.float32).reshape(-1)
            run_ret += r
            run_len += 1

            # 准备下一步
            obs = torch.as_tensor(next_obs_np, device=device)
            action_masks = self._norm_mask(info["action_mask"], B, A, device)

            # 读取刚刚完成的 episode（Gymnasium VectorEnv 会在 final_info 里给）
            if "final_info" in info:
                for i, fi in enumerate(info["final_info"]):
                    if fi is None:
                        continue

                    succ = _get_success(fi)
                    if fi.get("episode") is not None:
                        ep_r = float(fi["episode"]["r"])
                        ep_l = int(fi["episode"]["l"])
                    else:
                        # 没有 episode 统计就用兜底的累计
                        ep_r = float(run_ret[i])
                        ep_l = int(run_len[i])

                    if ptr < total_episodes:
                        wins[ptr]    = succ
                        returns[ptr] = ep_r
                        lengths[ptr] = ep_l
                        ptr += 1
                        finished += 1

                    # 清零该环境的在线累计，为下一局做准备
                    run_ret[i] = 0.0
                    run_len[i] = 0

            else:
                # 退化路径：没有 final_info，就用 done 标志 & info 里的 success 数组
                # （你的栈里大概率用不到这个分支）
                succ_arr = None
                for k in ("is_success", "success", "won", "win"):
                    if k in info:
                        succ_arr = np.asarray(info[k])
                        break
                for i, d in enumerate(done):
                    if not d:
                        continue
                    succ = bool(succ_arr[i]) if succ_arr is not None else False
                    ep_r = float(run_ret[i])
                    ep_l = int(run_len[i])
                    if ptr < total_episodes:
                        wins[ptr]    = succ
                        returns[ptr] = ep_r
                        lengths[ptr] = ep_l
                        ptr += 1
                        finished += 1
                    run_ret[i] = 0.0
                    run_len[i] = 0

            # 轻量进度打印（每完成 ~10 批并行 env 或收尾时）
            if finished and (finished % (10 * B) == 0 or finished >= total_episodes):
                print(f"[{prefix}] finished={finished}/{total_episodes}  "
                    f"win_rate={wins[:ptr].mean():.3f}")

        # 汇总
        out = {
            "win_rate": float(wins.mean()),
            "wins": wins,
            "returns": returns,
            "lengths": lengths,
        }
        print(f"[{prefix}] done. win_rate={out['win_rate']:.4f}, "
            f"avg_return={returns.mean():.3f}, avg_length={lengths.mean():.2f}")
        return out


    @torch.no_grad()
    def evaluate_video(self, prefix: str = "val"):
        cfg = self.config
        device = self.device
        env = self.val_env
        if self.video_wrapper_cls is not None and not isinstance(env, self.video_wrapper_cls):
            # assume already wrapped outside; keep it simple here
            pass
        obs, info = env.reset()
        B = 1
        A = self.envs.single_action_space.n
        next_obs = torch.as_tensor(obs, device=device).unsqueeze(0)
        # print(info["action_mask"].shape)
        action_masks = self._norm_mask(info["action_mask"], B, A, self.device)
        while True:
            action, probs, entropy  = self.model.get_action(next_obs, action_mask=action_masks, decode_type="greedy")
            obs2, reward, terminated, truncated, info = env.step(action.cpu().numpy(), probs=probs.squeeze().cpu().detach().numpy())
            action_masks = [info.get("action_mask")] if isinstance(info.get("action_mask"), np.ndarray) else info.get("action_mask")
            action_masks = self._norm_mask(info["action_mask"], B, A, self.device)
            done = np.logical_or(terminated, truncated)
            if done:
                break
            next_obs = torch.as_tensor(obs2, device=device).unsqueeze(0)
        time.sleep(0.5)
        print(f"Evaluation finished for {prefix}.")
        env.close()

    # ------------------------ training loop ------------------------
    def train(self):
        cfg = self.config
        start_time = time.time()
        num_updates = cfg.total_timesteps // cfg.batch_size

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                self.optimizer.param_groups[0]["lr"] = cfg.learning_rate * frac

            next_obs, extra = self.rollout()
            advantages, returns = self.compute_advantages(extra["next_done"], extra["next_value"])
            stats = self.update_policy(advantages, returns)

            # logging
            elapsed = time.time() - start_time
            progress = update / num_updates
            remaining = (elapsed / progress - elapsed) if progress > 0 else 0
            sps = int(self.state.global_step / max(1e-6, elapsed))

            print(
                f"Update {update}/{num_updates} | "
                f"Value Loss: {stats['v_loss']:.3f} | Policy Loss: {stats['pg_loss']:.3f} | "
                f"Entropy: {stats['entropy']:.3f} | Var: {stats['explained_var']:.3f} | "
                f"SPS: {sps} | Win Rate: {self.state.win_rate:.3f} | "
                f"Episodic Return: {self.state.episodic_return:.1f} | Episodic Length: {self.state.episodic_length:.1f} | "
                f"ETA: {format_seconds(remaining)}"
            )

            if self.writer is not None:
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.state.global_step)
                self.writer.add_scalar("losses/value_loss", stats['v_loss'], self.state.global_step)
                self.writer.add_scalar("losses/policy_loss", stats['pg_loss'], self.state.global_step)
                self.writer.add_scalar("losses/entropy", stats['entropy'], self.state.global_step)
                self.writer.add_scalar("losses/old_approx_kl", stats['old_approx_kl'], self.state.global_step)
                self.writer.add_scalar("losses/approx_kl", stats['approx_kl'], self.state.global_step)
                self.writer.add_scalar("losses/clipfrac", stats['clipfrac'], self.state.global_step)
                self.writer.add_scalar("losses/explained_variance", stats['explained_var'], self.state.global_step)
                self.writer.add_scalar("charts/SPS", sps, self.state.global_step)
                self.writer.add_scalar("charts/win_rate", self.state.win_rate, self.state.global_step)
                self.writer.add_scalar("charts/episodic_return", self.state.episodic_return, self.state.global_step)
                self.writer.add_scalar("charts/episodic_length", self.state.episodic_length, self.state.global_step)
                

            self.maybe_eval_and_save(update)
        
    # ------------------------ helpers ------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def _norm_mask(self, action_mask, B: int, A: int, device):
        if isinstance(action_mask, torch.Tensor):
            m = action_mask.to(device=device, dtype=torch.bool)
        else:
            arr = action_mask
            if isinstance(arr, (list, tuple)):
                arr = np.array([np.asarray(m, dtype=bool).reshape(-1) for m in arr], dtype=bool)
            else:
                arr = np.asarray(arr)
                if arr.dtype == object:
                    arr = np.vstack([np.asarray(m, dtype=bool).reshape(-1) for m in arr])
                else:
                    arr = arr.astype(bool)
            m = torch.as_tensor(arr, dtype=torch.bool, device=device)
        if m.ndim == 1:
            m = m.unsqueeze(0).expand(B, -1)
        assert m.shape == (B, A), f"mask shape {m.shape} != {(B,A)}"
        return m