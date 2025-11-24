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


class CE:
    """
    Encapsulates CROSS ENTROPY training/evaluation for MinesweeperModel.
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
        H, W = shape_obs
        self.obs      = torch.zeros((config.num_steps, config.num_envs) + shape_obs, device=device)
        self.actions  = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape, device=device)
        self.logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.values   = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.rewards  = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.dones    = torch.zeros((config.num_steps, config.num_envs), device=device)
        
        A = envs.single_action_space.n
        self.act_masks   = torch.zeros((config.num_steps, config.num_envs, A), dtype=torch.bool, device=device)
        self.safe_local  = torch.zeros((config.num_steps, config.num_envs, A), dtype=torch.bool, device=device)
        self.safe_global = torch.zeros((config.num_steps, config.num_envs, A), dtype=torch.bool, device=device)


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
        
        curr_safe_local, curr_safe_global = self._extract_safe_masks(info, B, A, self.device)
        
        H, W = self.envs.single_observation_space.shape
        # curr_full = self._stack_full_board(info["full_board"], B, H, W)  # NEW
        curr_full = None
        
        next_obs = torch.as_tensor(obs_np, device=device)

        final_info_seen = 0
        total_win = 0
        total_return = 0.0
        total_length = 0

        for step in range(cfg.num_steps):
            self.state.global_step += cfg.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done
            
            self.act_masks[step]   = action_masks
            self.safe_local[step]  = curr_safe_local
            self.safe_global[step] = curr_safe_global

            with torch.no_grad():
                action, logprob, _, value = self.model.get_action_and_value(
                    next_obs, action_mask=action_masks, full_board=curr_full,  # NEW
                )
                self.values[step] = value.flatten()

            self.actions[step] = action
            self.logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            # curr_full = self._stack_full_board(info["full_board"], B, H, W)  # NEW
            # self.full_boards[step] = curr_full  # NEW
            action_masks = self._norm_mask(info["action_mask"], B, A, self.device)
            
            curr_safe_local, curr_safe_global = self._extract_safe_masks(info, B, A, self.device)
            
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
            next_value = self.model.get_value(
                next_obs, full_board=curr_full  # NEW
            ).reshape(1, -1)

        out = {
            "next_obs": next_obs,
            "next_done": next_done,
            "next_value": next_value,
        }
        return out["next_obs"], out

    def update_policy(self):
        cfg = self.config

        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)

        b_inds = np.arange(cfg.batch_size)

        H, W = self.envs.single_observation_space.shape

        for epoch in range(cfg.update_epoches):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                mb_inds = b_inds[start:end]

                # ===== 监督项：soft cross-entropy over safe sets =====
                A = self.envs.single_action_space.n
                b_amask = self.act_masks.reshape(-1, A)
                b_sloc  = self.safe_local.reshape(-1, A)
                b_sglob = self.safe_global.reshape(-1, A)

                mb_amask = b_amask[mb_inds]
                mb_sloc  = b_sloc[mb_inds]
                mb_sglob = (b_sglob[mb_inds] & ~mb_sloc)  # 去重

                # 取当前策略 logits（policy 不看 full_board）
                mb_out = self.model.forward(b_obs[mb_inds], full_board=None)
                mb_logits = mb_out[0]
                # 屏蔽非法动作
                mb_logits = mb_logits.masked_fill(~mb_amask, torch.finfo(mb_logits.dtype).min)

                w_local  = getattr(cfg, "w_local", 1.0)
                w_global = getattr(cfg, "w_global", 0.3)

                target = torch.zeros_like(mb_logits, dtype=mb_logits.dtype)
                if w_local > 0:
                    cntL = mb_sloc.sum(dim=1, keepdim=True).clamp_min(1)
                    target = target + w_local * (mb_sloc.float() / cntL)
                if w_global > 0:
                    cntG = mb_sglob.sum(dim=1, keepdim=True).clamp_min(1)
                    target = target + w_global * (mb_sglob.float() / cntG)

                sumw = target.sum(dim=1, keepdim=True)
                has_label = (sumw.squeeze(1) > 0).float()            # 有监督标签的样本
                # 只对有标签的样本做归一化
                safe_target = torch.where(sumw > 0, target / sumw, target)

                logp = torch.log_softmax(mb_logits, dim=-1)
                sup_loss_all = -(safe_target * logp).sum(dim=1)
                loss = (sup_loss_all * has_label).sum() / has_label.sum().clamp_min(1)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

        stats = {
            "loss": loss.item(),
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
        out = self.evaluate_n_episodes(
            total_episodes=cfg.num_envs * 10, prefix=f"test_{self.state.global_step}"
        )
        if self.writer is not None:
            self.writer.add_scalar("eval/win_rate", out["win_rate"], self.state.global_step)
            self.writer.add_scalar("eval/avg_return", out["returns"].mean(), self.state.global_step)
            self.writer.add_scalar("eval/avg_length", out["lengths"].mean(), self.state.global_step)

    # ------------------------ training loop ------------------------
    def train(self):
        cfg = self.config
        start_time = time.time()
        num_updates = cfg.total_timesteps // cfg.batch_size

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                # 若配置中提供了自定义学习率调度表，则优先使用；
                # 否则回退到与 PPO 相同的线性衰减策略。
                schedule_ts = getattr(cfg, "schedule_timesteps", None)
                schedule_lr = getattr(cfg, "schedule_lr", None)
                if schedule_ts is not None and schedule_lr is not None:
                    progress = update * cfg.batch_size / cfg.total_timesteps
                    idx = 0
                    while idx < len(schedule_ts) and progress > schedule_ts[idx]:
                        idx += 1
                    if idx >= len(schedule_lr):
                        idx = len(schedule_lr) - 1
                    frac = float(schedule_lr[idx])
                else:
                    frac = 1.0 - (update - 1.0) / max(1, num_updates)
                self.optimizer.param_groups[0]["lr"] = cfg.learning_rate * frac

            next_obs, extra = self.rollout()
            stats = self.update_policy()

            # logging
            elapsed = time.time() - start_time
            progress = update / num_updates
            remaining = (elapsed / progress - elapsed) if progress > 0 else 0
            sps = int(self.state.global_step / max(1e-6, elapsed))

            print(
                f"Update {update}/{num_updates} | "
                f"Loss: {stats['loss']:.3f} | "
                f"SPS: {sps} | Win Rate: {self.state.win_rate:.3f} | "
                f"ETA: {format_seconds(remaining)}"
            )

            if self.writer is not None:
                self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], self.state.global_step)
                self.writer.add_scalar("train/loss", stats["loss"], self.state.global_step)
                self.writer.add_scalar("train/SPS", sps, self.state.global_step)
                self.writer.add_scalar("train/win_rate", self.state.win_rate, self.state.global_step)
                
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
    
    def _extract_safe_masks(self, info, B: int, A: int, device):
        """
        从 info 中提取 local/global 安全集合，统一转成 [B, A] 的 bool mask。
        任何缺失/None/空输入都会返回全 0（表示该步没有监督标签）。
        支持以下形式：
        - mask：list[list[bool]] 或 ndarray[bool] 形状 [B, A]
        - 索引：list[list[int]]（每个子列表是动作索引）
        - 单一安全动作：list[int] 或 ndarray[int] 形状 [B]（每个 env 一个索引）
        """

        def zeros():
            return torch.zeros((B, A), dtype=torch.bool, device=device)

        if info is None:
            return zeros(), zeros()

        import numpy as _np

        def _to_mask(x):
            # 任何 None / 空，都视为无标签
            if x is None:
                return zeros()
            # 允许 info 是标量 / 单索引 / [B] 形式（单一安全动作）
            if isinstance(x, (list, tuple, _np.ndarray)) and _np.array(x, dtype=object).ndim == 1:
                arr = _np.array(x, dtype=object)
                # 若是 [B] 的整型索引
                if arr.size == B and all((xi is None) or isinstance(xi, (int, _np.integer)) for xi in arr):
                    m = _np.zeros((B, A), dtype=bool)
                    for i, idx in enumerate(arr):
                        if idx is not None and 0 <= int(idx) < A:
                            m[i, int(idx)] = True
                    return torch.as_tensor(m, dtype=torch.bool, device=device)

            # 常规：mask 或 索引列表
            try:
                arr = _np.asarray(x)
                # 直接是 bool mask
                if arr.dtype == bool or arr.dtype == _np.bool_:
                    arr = arr.reshape(B, A)
                    return torch.as_tensor(arr, dtype=torch.bool, device=device)
                # 对象数组里是 per-env mask 或索引
                if arr.dtype == object:
                    # 优先尝试把每个元素当 mask
                    try:
                        m = _np.vstack([_np.asarray(m, dtype=bool).reshape(-1) for m in arr])
                        m = m.reshape(B, A)
                        return torch.as_tensor(m, dtype=torch.bool, device=device)
                    except Exception:
                        pass
                    # 再尝试当索引列表
                    m = _np.zeros((B, A), dtype=bool)
                    for i, idxs in enumerate(arr):
                        if idxs is None:
                            continue
                        idxs = _np.asarray(idxs, dtype=int).reshape(-1)
                        idxs = idxs[(idxs >= 0) & (idxs < A)]
                        m[i, idxs] = True
                    return torch.as_tensor(m, dtype=torch.bool, device=device)
                # 其它数值数组：当作索引列表（展平）
                arr = arr.astype(int).reshape(-1)
                m = _np.zeros((B, A), dtype=bool)
                # 如果长度等于 B，按“每 env 一个单索引”
                if arr.size == B:
                    for i, idx in enumerate(arr):
                        if 0 <= int(idx) < A:
                            m[i, int(idx)] = True
                else:
                    # 否则统一给所有 env（很少用到）
                    idxs = arr[(arr >= 0) & (arr < A)]
                    m[:, idxs] = True
                return torch.as_tensor(m, dtype=torch.bool, device=device)
            except Exception:
                return zeros()

        # 兼容多键名；都没有就返回全 0
        L = None
        G = None
        for k in ("safe_local_mask", "safe_local", "local_safe", "safe_mask_local"):
            if k in info:
                L = info[k]
                break
        for k in ("safe_global_mask", "safe_global", "global_safe", "safe_mask_global"):
            if k in info:
                G = info[k]
                break
        if L is None and G is None:
            for k in ("safe_mask", "safe_actions", "safe"):
                if k in info:
                    L = info[k]
                    break

        mL = _to_mask(L)
        mG = _to_mask(G)
        # 去重，避免同一动作既在 local 又在 global
        mG = mG & (~mL)
        return mL, mG
