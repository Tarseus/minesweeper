# ==============================
# file: models/minesweeper_model.py
# ==============================
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class SqueezeExcite(nn.Module):
    """Channel attention (SE)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg_pool(x))
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SqueezeExcite(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return self.relu(x + y)


class CNNBased(nn.Module):
    """
    Convolutional policy/value network for Minesweeper.
    - Input: one-hot 11 channels board (B,11,H,W) from int board (B,H,W) in [0..10]
    - Policy head outputs logits over H*W actions
    - Value head can (optionally) use privileged full_board
    """
    def __init__(self, obs_shape: Tuple[int, int], stop_actor_grad_in_value: bool = True):
        super().__init__()
        H, W = obs_shape
        self.H, self.W = H, W
        self.board_size = H * W
        in_ch = 11

        # ----- policy/observation backbone -----
        obs_layers = [
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        for _ in range(7):
            obs_layers.append(ResBlock(64))
        self.obs_backbone = nn.Sequential(*obs_layers)

        # ----- privileged full-board encoder (只给 value 用) -----
        # 轻一点即可；你也可以用更深的结构
        self.full_encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            ResBlock(32),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # policy head 只接 obs 特征
        self.policy_head = nn.Conv2d(64, 1, kernel_size=1)  # (B,1,H,W)

        # value head 融合 obs_vec(64) 与 full_vec(32) -> 96
        self.value_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.stop_actor_grad_in_value = stop_actor_grad_in_value
        self.full_vec_dim = 32

    @staticmethod
    def _one_hot(obs: torch.Tensor) -> torch.Tensor:
        """(B,H,W) -> (B,11,H,W) float"""
        return F.one_hot(obs.long(), num_classes=11).permute(0, 3, 1, 2).float()

    def forward(
        self,
        x: torch.Tensor,
        full_board: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ----- policy path -----
        obs_oh   = self._one_hot(x)
        obs_feat = self.obs_backbone(obs_oh)                # (B,64,H,W)
        logits   = self.policy_head(obs_feat).flatten(1)    # (B,H*W)

        # ----- value path -----
        obs_vec = self.pool(obs_feat).flatten(1)            # (B,64)
        if self.stop_actor_grad_in_value:
            obs_vec = obs_vec.detach()  # 阻断 value 损失对 actor 主干的梯度

        if full_board is not None:
            full_oh   = self._one_hot(full_board)           # (B,11,H,W)，含地雷与周围数的全真值
            full_feat = self.full_encoder(full_oh)           # (B,32,H,W)
            full_vec  = self.pool(full_feat).flatten(1)      # (B,32)
        else:
            # 推理或无权限时，给零向量占位（保证形状恒定）
            B = x.shape[0]
            full_vec = obs_vec.new_zeros(B, self.full_vec_dim)

        v_in  = torch.cat([obs_vec, full_vec], dim=1)        # (B,96)
        value = self.value_head(v_in)                        # (B,1)

        return logits, value

    def get_value(self, x: torch.Tensor, full_board: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, v = self.forward(x, full_board=full_board)
        return v

    def get_action(
        self,
        x: torch.Tensor,
        action_mask: Optional[object] = None,
    ):
        device = x.device
        logits, _ = self.forward(x, full_board=None)  # policy 从不看 full_board

        # mask illegal actions
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError("action_mask must be a torch.BoolTensor shaped (B,A) after refactor")
            mask = action_mask
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        probs  = Categorical(logits=logits)
        action = probs.sample()
        return action, probs, probs.entropy()

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[object] = None,
        full_board: Optional[torch.Tensor] = None,
    ):
        device = x.device
        logits, value = self.forward(x, full_board=full_board)  # 这里把全局真值喂给 value

        # mask illegal actions
        # if action_mask is not None:
        #     mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        #     if mask.ndim == 1:
        #         mask = mask.unsqueeze(0)
        #     if mask.shape[0] != logits.shape[0]:
        #         try:
        #             import numpy as _np
        #             mask = torch.as_tensor(_np.vstack(action_mask), dtype=torch.bool, device=device)
        #         except Exception:
        #             raise ValueError("action_mask shape mismatch; expected (B,H*W)")
        #     logits = logits.masked_fill(~mask, -1e20)
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError("action_mask must be a torch.BoolTensor shaped (B,A) after refactor")
            mask = action_mask
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        elif not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.long, device=device)

        return action, probs.log_prob(action), probs.entropy(), value