from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


_DIR8: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@lru_cache(maxsize=128)
def _build_grid_edges_8n_cpu(h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build directed 8-neighborhood edges for an h x w grid.

    Returns CPU tensors:
      - edge_src: (E,)
      - edge_dst: (E,)
      - edge_type: (E,) in [0..7] matching _DIR8
    Node index is row-major: idx = r*w + c
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid grid size: {(h, w)}")

    idx = torch.arange(h * w, dtype=torch.long).reshape(h, w)
    src_all = []
    dst_all = []
    typ_all = []

    for t, (dr, dc) in enumerate(_DIR8):
        r0 = max(0, -dr)
        r1 = h - max(0, dr)
        c0 = max(0, -dc)
        c1 = w - max(0, dc)

        src = idx[r0:r1, c0:c1]
        dst = idx[r0 + dr : r1 + dr, c0 + dc : c1 + dc]

        src_all.append(src.reshape(-1))
        dst_all.append(dst.reshape(-1))
        typ_all.append(torch.full((src.numel(),), t, dtype=torch.long))

    edge_src = torch.cat(src_all, dim=0)
    edge_dst = torch.cat(dst_all, dim=0)
    edge_type = torch.cat(typ_all, dim=0)
    return edge_src, edge_dst, edge_type


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MessagePassingBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_edge: int,
        d_msg: Optional[int] = None,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_msg = int(d_msg or d_model)
        self.d_model = d_model
        self.d_msg = d_msg

        self.msg_mlp = _MLP(2 * d_model + d_edge, mlp_hidden, d_msg, dropout=dropout)
        self.gru = nn.GRUCell(input_size=d_msg, hidden_size=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,  # (B,N,d_model)
        edge_src: torch.Tensor,  # (E,)
        edge_dst: torch.Tensor,  # (E,)
        edge_emb: torch.Tensor,  # (E,d_edge)
    ) -> torch.Tensor:
        bsz, n_nodes, _ = h.shape
        e = edge_src.numel()

        h_src = h.index_select(1, edge_src)  # (B,E,d)
        h_dst = h.index_select(1, edge_dst)  # (B,E,d)
        e_emb = edge_emb.unsqueeze(0).expand(bsz, e, -1)  # (B,E,de)

        m_in = torch.cat([h_src, h_dst, e_emb], dim=-1)  # (B,E,2d+de)
        m = self.msg_mlp(m_in.reshape(bsz * e, -1)).reshape(bsz, e, self.d_msg)  # (B,E,d_msg)

        agg = h.new_zeros((bsz, n_nodes, self.d_msg))  # (B,N,d_msg)
        agg.index_add_(1, edge_dst, m)

        h_next = self.gru(
            agg.reshape(bsz * n_nodes, self.d_msg),
            h.reshape(bsz * n_nodes, self.d_model),
        ).reshape(bsz, n_nodes, self.d_model)
        return self.norm(h_next)


class GridGNNBased(nn.Module):
    """
    Size-agnostic Grid-GNN actor-critic for Minesweeper.

    - Input: (B,H,W) int board with values in [0..10] (0-8 numbers, 9 mine, 10 unknown)
    - Output:
        logits: (B,H*W) per-cell action logits (open that cell)
        value:  (B,1) state value
    """

    def __init__(
        self,
        obs_shape: Optional[Tuple[int, int]] = None,
        d_model: int = 128,
        n_rounds: int = 12,
        edge_emb_dim: int = 16,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
        stop_actor_grad_in_value: bool = True,
    ):
        super().__init__()
        self.stop_actor_grad_in_value = stop_actor_grad_in_value
        self.d_model = d_model

        in_dim = 11  # one-hot for values [0..10]
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.edge_emb = nn.Embedding(8, edge_emb_dim)
        self.blocks = nn.ModuleList(
            [
                MessagePassingBlock(
                    d_model=d_model,
                    d_edge=edge_emb_dim,
                    d_msg=d_model,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                )
                for _ in range(n_rounds)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        self._graph_cache: Dict[Tuple[int, int, str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        if obs_shape is not None:
            h, w = obs_shape
            edge_src, edge_dst, edge_type = _build_grid_edges_8n_cpu(h, w)
            self.register_buffer("_edge_src", edge_src, persistent=False)
            self.register_buffer("_edge_dst", edge_dst, persistent=False)
            self.register_buffer("_edge_type", edge_type, persistent=False)
        else:
            self._edge_src = None
            self._edge_dst = None
            self._edge_type = None

    @staticmethod
    def _one_hot(x: torch.Tensor) -> torch.Tensor:
        x = x.long().clamp(0, 10)
        return F.one_hot(x, num_classes=11).float()

    def _get_graph(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._edge_src is not None:
            return self._edge_src.to(device), self._edge_dst.to(device), self._edge_type.to(device)

        dev_type = device.type
        dev_index = -1 if device.index is None else int(device.index)
        key = (h, w, dev_type, dev_index)
        cached = self._graph_cache.get(key)
        if cached is not None:
            return cached

        edge_src, edge_dst, edge_type = _build_grid_edges_8n_cpu(h, w)
        edge_src = edge_src.to(device)
        edge_dst = edge_dst.to(device)
        edge_type = edge_type.to(device)
        self._graph_cache[key] = (edge_src, edge_dst, edge_type)
        return edge_src, edge_dst, edge_type

    def forward(self, x: torch.Tensor, full_board: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, h, w = x.shape
        n_nodes = h * w

        x_oh = self._one_hot(x).reshape(bsz, n_nodes, -1)  # (B,N,11)
        h0 = self.in_proj(x_oh)  # (B,N,d)

        edge_src, edge_dst, edge_type = self._get_graph(h, w, x.device)
        edge_emb = self.edge_emb(edge_type)  # (E,de)

        h_state = h0
        for blk in self.blocks:
            h_state = blk(h_state, edge_src=edge_src, edge_dst=edge_dst, edge_emb=edge_emb)

        logits = self.policy_head(h_state).squeeze(-1)  # (B,N)

        if self.stop_actor_grad_in_value:
            h_val = h_state.detach()
        else:
            h_val = h_state

        mean_pool = h_val.mean(dim=1)
        max_pool = h_val.max(dim=1).values
        value = self.value_head(torch.cat([mean_pool, max_pool], dim=-1))  # (B,1)
        return logits, value

    def get_value(self, x: torch.Tensor, full_board: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, v = self.forward(x, full_board=full_board)
        return v

    def get_action(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        decode_type: str = "sample",
    ):
        logits, _ = self.forward(x, full_board=None)
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError("action_mask must be a torch.BoolTensor shaped (B,A)")
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        dist = Categorical(logits=logits)
        if decode_type == "greedy":
            action = torch.argmax(dist.probs, dim=1)
        else:
            action = dist.sample()
        return action, dist.probs, dist.entropy()

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        full_board: Optional[torch.Tensor] = None,
    ):
        device = x.device
        logits, value = self.forward(x, full_board=full_board)

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError("action_mask must be a torch.BoolTensor shaped (B,A)")
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        elif not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.long, device=device)

        return action, dist.log_prob(action), dist.entropy(), value
