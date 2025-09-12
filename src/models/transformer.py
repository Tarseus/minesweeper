from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with self-attention and MLP."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-LN architecture
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class TransformerBasedModel(nn.Module):
    """
    Transformer-based policy/value network for Minesweeper following the design in transformer.md.
    - Uses a decoder-only transformer with global tokens
    - Board tokens + global tokens with dynamic updating
    - Pointer mechanism for action selection
    - Value and mine probability heads
    """
    def __init__(
        self, 
        obs_shape: Tuple[int, int],
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        n_global_tokens: int = 8,
        dropout: float = 0.1,
        stop_actor_grad_in_value: bool = True,
    ):
        super().__init__()
        H, W = obs_shape
        self.H, self.W = H, W
        self.board_size = H * W
        self.d_model = d_model
        self.n_global_tokens = n_global_tokens
        self.stop_actor_grad_in_value = stop_actor_grad_in_value
        
        # Discrete embeddings
        self.tile_embed = nn.Embedding(12, d_model)  # 0-8: numbers, 9: mine, 10: unknown, 11: flag
        self.row_embed = nn.Embedding(H, d_model)
        self.col_embed = nn.Embedding(W, d_model)
        
        # Numerical feature projection
        self.feat_proj = nn.Linear(6, d_model)  # 6 numerical features
        
        # Global tokens (learnable)
        self.global_tokens = nn.Parameter(torch.randn(1, n_global_tokens, d_model) * 0.02)
        
        # Optional 2D relative position bias
        self.use_rel_pos_bias = False
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Policy head (pointer network)
        self.pointer_key = nn.Linear(d_model, d_model)
        self.pointer_query = nn.Linear(d_model, d_model)
        self.pointer_score = nn.Linear(d_model, 1)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Mine probability head
        self.mine_prob_head = nn.Linear(d_model, 1)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract numerical features from the board state.
        x: (B,H,W) tensor with values 0-11
        Returns: (B,H*W,6) tensor with numerical features
        """
        B, H, W = x.shape
        device = x.device
        features = torch.zeros(B, H, W, 6, device=device)
        
        # Extract features as described in transformer.md
        is_unknown = (x == 10)
        is_flag = (x == 11)
        is_number = (x >= 0) & (x <= 8)
        digit_value = x.clamp(0, 8) / 8.0
        
        # Calculate adjacent unknowns and flags using convolution
        kernel = torch.tensor([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        # Pad and convolve
        unknown_padded = F.pad(is_unknown.float(), (1, 1, 1, 1))
        flag_padded = F.pad(is_flag.float(), (1, 1, 1, 1))
        
        adj_unknown = F.conv2d(unknown_padded.view(B, 1, H+2, W+2), kernel) / 8.0
        adj_flag = F.conv2d(flag_padded.view(B, 1, H+2, W+2), kernel) / 8.0
        
        adj_unknown = adj_unknown.view(B, H, W)
        adj_flag = adj_flag.view(B, H, W)
        
        # Frontier: unknown cells adjacent to numbers
        number_padded = F.pad(is_number.float(), (1, 1, 1, 1))
        adj_number = F.conv2d(number_padded.view(B, 1, H+2, W+2), kernel).view(B, H, W) > 0
        frontier = is_unknown & adj_number
        
        # Residual: max(0, digit - adj_flags)/8
        residual = (x.clamp(0, 8) - (adj_flag * 8).floor().clamp(0, 8)).clamp(0, 8) / 8.0
        residual = residual * is_number.float()
        
        # Assemble features
        features[..., 0] = frontier.float()
        features[..., 1] = adj_unknown
        features[..., 2] = adj_flag
        features[..., 3] = is_number.float()
        features[..., 4] = digit_value
        features[..., 5] = residual
        
        return features.view(B, H*W, 6)
    
    def forward(
        self,
        x: torch.Tensor,
        full_board: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer.
        
        Args:
            x: (B,H,W) tensor with values 0-11
            full_board: Optional ground truth board (for privileged value function)
            
        Returns:
            logits: (B,H*W) action logits
            value: (B,1) value prediction
            mine_probs: (B,H*W) mine probability predictions
        """
        B = x.shape[0]
        device = x.device
        N = self.H * self.W
        
        # Extract features
        numerical_features = self.extract_features(x)
        
        # Create position indices
        row_indices = torch.arange(self.H, device=device).view(1, self.H, 1).expand(B, self.H, self.W).reshape(B, N)
        col_indices = torch.arange(self.W, device=device).view(1, 1, self.W).expand(B, self.H, self.W).reshape(B, N)
        
        # Get embeddings
        tile_ids = x.reshape(B, N)
        tile_emb = self.tile_embed(tile_ids.int())
        row_emb = self.row_embed(row_indices)
        col_emb = self.col_embed(col_indices)
        feat_emb = self.feat_proj(numerical_features)
        
        # Combine embeddings
        cell_tokens = tile_emb + row_emb + col_emb + feat_emb
        
        # Append global tokens
        global_tokens = self.global_tokens.expand(B, -1, -1)
        tokens = torch.cat([cell_tokens, global_tokens], dim=1)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        
        # Apply final layer norm
        tokens = self.layer_norm(tokens)
        
        # Split back into cell tokens and global tokens
        cell_tokens = tokens[:, :N]
        global_tokens = tokens[:, N:]
        
        # Global context for value and pointer
        global_ctx = global_tokens.mean(dim=1)
        
        # Compute policy logits (pointer mechanism)
        cell_keys = self.pointer_key(cell_tokens)
        global_query = self.pointer_query(global_ctx).unsqueeze(1)
        
        # Compute attention scores
        attn = torch.matmul(global_query, cell_keys.transpose(1, 2)) / (self.d_model ** 0.5)
        logits = attn.squeeze(1)
        
        # Compute value
        if self.stop_actor_grad_in_value:
            cell_mean = cell_tokens.detach().mean(dim=1)
            global_ctx_detached = global_ctx.detach()
        else:
            cell_mean = cell_tokens.mean(dim=1)
            global_ctx_detached = global_ctx
            
        value_input = torch.cat([cell_mean, global_ctx_detached], dim=1)
        value = self.value_head(value_input)
        
        # Compute mine probabilities
        mine_logits = self.mine_prob_head(cell_tokens).squeeze(-1)
        mine_probs = torch.sigmoid(mine_logits)
        
        return logits, value, mine_probs
    
    def get_value(self, x: torch.Tensor, full_board: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get value prediction for the given state."""
        _, value, _ = self.forward(x, full_board)
        return value
    
    def get_action(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        decode_type: str = "sample",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            x: (B,H,W) tensor with values 0-11
            action_mask: (B,H*W) boolean tensor indicating valid actions
            decode_type: "sample" or "greedy"
            
        Returns:
            action: (B,) tensor with selected action indices
            probs: (B,H*W) tensor with action probabilities
            entropy: (B,) tensor with entropy of policy distribution
        """
        logits, _, _ = self.forward(x, full_board=None)
        
        # Mask invalid actions
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError("action_mask must be a torch.BoolTensor shaped (B,A) after refactor")
            logits = logits.masked_fill(~action_mask, -float('inf'))
        
        probs = Categorical(logits=logits)
        if decode_type == "greedy":
            action = torch.argmax(probs.probs, dim=1)
        else:
            action = probs.sample()
            
        return action, probs.probs, probs.entropy()
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        full_board: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value for the given state.
        
        Args:
            x: (B,H,W) tensor with values 0-11
            action: (B,) tensor with action indices (optional)
            action_mask: (B,H*W) boolean tensor indicating valid actions
            full_board: Optional ground truth board for privileged value function
            
        Returns:
            action: (B,) tensor with selected action indices
            log_prob: (B,) tensor with log probabilities of selected actions
            entropy: (B,) tensor with entropy of policy distribution
            value: (B,1) tensor with value prediction
        """
        device = x.device
        logits, value, _ = self.forward(x, full_board=full_board)
        
        # Mask invalid actions
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError("action_mask must be a torch.BoolTensor shaped (B,A) after refactor")
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        elif not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.long, device=device)
        
        return action, probs.log_prob(action), probs.entropy(), value
    
    def get_mine_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get mine probabilities for all cells."""
        _, _, mine_probs = self.forward(x)
        return mine_probs