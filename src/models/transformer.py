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


class EncoderBlock(nn.Module):
    """Pre-LN encoder block with self-attention and MLP."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        return x


class DecoderBlock(nn.Module):
    """Pre-LN decoder block with self-attention, cross-attention and MLP."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention over decoder tokens
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, key_padding_mask=self_key_padding_mask
        )
        x = x + self.dropout(self_attn_out)

        # Cross-attention: decoder queries, encoder keys/values
        x_norm = self.norm2(x)
        cross_out, _ = self.cross_attn(
            x_norm,
            encoder_output,
            encoder_output,
            key_padding_mask=encoder_key_padding_mask,
        )
        x = x + self.dropout(cross_out)

        # Feed-forward
        x_norm = self.norm3(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        return x


class TransformerEncoderDecoderModel(nn.Module):
    """
    Encoder-decoder Transformer policy/value network for Minesweeper.

    - Encoder operates on opened cells (and encoder global tokens)
    - Decoder operates on unopened cells (and decoder global tokens)
    - Decoder uses cross-attention to read from encoder outputs
    - Pointer-style policy head over all board cells
    - Value and mine-probability heads
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int],
        d_model: int = 384,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        n_encoder_global_tokens: int = 4,
        n_decoder_global_tokens: int = 4,
        dropout: float = 0.1,
        stop_actor_grad_in_value: bool = True,
    ):
        super().__init__()
        H, W = obs_shape
        self.H, self.W = H, W
        self.board_size = H * W
        self.d_model = d_model
        self.stop_actor_grad_in_value = stop_actor_grad_in_value

        self.n_encoder_global_tokens = n_encoder_global_tokens
        self.n_decoder_global_tokens = n_decoder_global_tokens

        # Discrete embeddings as in TransformerBasedModel
        self.tile_embed = nn.Embedding(12, d_model)
        self.row_embed = nn.Embedding(H, d_model)
        self.col_embed = nn.Embedding(W, d_model)

        # Numerical feature projection (6 features per cell)
        self.feat_proj = nn.Linear(6, d_model)

        # Global tokens
        self.encoder_global_tokens = nn.Parameter(
            torch.randn(1, n_encoder_global_tokens, d_model) * 0.02
        )
        self.decoder_global_tokens = nn.Parameter(
            torch.randn(1, n_decoder_global_tokens, d_model) * 0.02
        )

        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, dropout=dropout) for _ in range(n_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, dropout=dropout) for _ in range(n_decoder_layers)]
        )

        # Output normalization
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Policy head (pointer over all cells)
        self.pointer_key = nn.Linear(d_model, d_model)
        self.pointer_query = nn.Linear(d_model, d_model)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Mine probability head
        self.mine_prob_head = nn.Linear(d_model, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Same numerical feature extraction as TransformerBasedModel.

        x: (B,H,W) tensor with values 0-11
        Returns: (B,H*W,6) tensor with numerical features
        """
        B, H, W = x.shape
        device = x.device
        features = torch.zeros(B, H, W, 6, device=device)

        is_unknown = x == 10
        is_flag = x == 11
        is_number = (x >= 0) & (x <= 8)
        digit_value = x.clamp(0, 8) / 8.0

        kernel = torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)

        unknown_padded = F.pad(is_unknown.float(), (1, 1, 1, 1))
        flag_padded = F.pad(is_flag.float(), (1, 1, 1, 1))

        adj_unknown = F.conv2d(unknown_padded.view(B, 1, H + 2, W + 2), kernel) / 8.0
        adj_flag = F.conv2d(flag_padded.view(B, 1, H + 2, W + 2), kernel) / 8.0

        adj_unknown = adj_unknown.view(B, H, W)
        adj_flag = adj_flag.view(B, H, W)

        number_padded = F.pad(is_number.float(), (1, 1, 1, 1))
        adj_number = F.conv2d(number_padded.view(B, 1, H + 2, W + 2), kernel).view(
            B, H, W
        ) > 0
        frontier = is_unknown & adj_number

        residual = (
            x.clamp(0, 8) - (adj_flag * 8).floor().clamp(0, 8)
        ).clamp(0, 8) / 8.0
        residual = residual * is_number.float()

        features[..., 0] = frontier.float()
        features[..., 1] = adj_unknown
        features[..., 2] = adj_flag
        features[..., 3] = is_number.float()
        features[..., 4] = digit_value
        features[..., 5] = residual

        return features.view(B, H * W, 6)

    def _embed_board(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-cell embeddings (without separating known/unknown).
        Returns: (B,N,d_model)
        """
        B = x.shape[0]
        device = x.device
        N = self.H * self.W

        numerical_features = self.extract_features(x)

        row_indices = (
            torch.arange(self.H, device=device)
            .view(1, self.H, 1)
            .expand(B, self.H, self.W)
            .reshape(B, N)
        )
        col_indices = (
            torch.arange(self.W, device=device)
            .view(1, 1, self.W)
            .expand(B, self.H, self.W)
            .reshape(B, N)
        )

        tile_ids = x.reshape(B, N)
        tile_emb = self.tile_embed(tile_ids.int())
        row_emb = self.row_embed(row_indices)
        col_emb = self.col_embed(col_indices)
        feat_emb = self.feat_proj(numerical_features)

        cell_tokens = tile_emb + row_emb + col_emb + feat_emb
        return cell_tokens

    def _build_encoder_inputs(
        self, cell_tokens: torch.Tensor, unknown_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build encoder input tokens and padding mask from cell tokens.

        Encoder sees opened cells (x != 10) plus encoder global tokens.
        """
        B, N, d = cell_tokens.shape
        device = cell_tokens.device

        known_mask = ~unknown_mask
        encoder_cell_tokens = []
        lengths = []
        for b in range(B):
            tokens_b = cell_tokens[b, known_mask[b]]
            encoder_cell_tokens.append(tokens_b)
            lengths.append(tokens_b.shape[0])

        max_cells = max(lengths) if lengths else 0
        seq_len = max_cells + self.n_encoder_global_tokens

        encoder_global = self.encoder_global_tokens.expand(B, -1, -1)
        encoder_tokens = cell_tokens.new_zeros(B, seq_len, d)
        key_padding_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

        for b in range(B):
            l = lengths[b]
            if l > 0:
                encoder_tokens[b, :l] = encoder_cell_tokens[b]
            if max_cells > l:
                key_padding_mask[b, l:max_cells] = True
            encoder_tokens[b, max_cells : max_cells + self.n_encoder_global_tokens] = (
                encoder_global[b]
            )

        return encoder_tokens, key_padding_mask

    def _build_decoder_inputs(
        self, cell_tokens: torch.Tensor, unknown_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build decoder input tokens, key padding mask, and index mapping
        for unknown cells.

        Decoder operates on unopened cells (x == 10) plus decoder global tokens.
        """
        B, N, d = cell_tokens.shape
        device = cell_tokens.device

        decoder_cell_tokens = []
        lengths = []
        unknown_indices = []
        for b in range(B):
            mask_b = unknown_mask[b]
            idx_b = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
            unknown_indices.append(idx_b)
            tokens_b = cell_tokens[b, mask_b]
            decoder_cell_tokens.append(tokens_b)
            lengths.append(tokens_b.shape[0])

        max_cells = max(lengths) if lengths else 0
        seq_len = max_cells + self.n_decoder_global_tokens

        decoder_global = self.decoder_global_tokens.expand(B, -1, -1)
        decoder_tokens = cell_tokens.new_zeros(B, seq_len, d)
        key_padding_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

        for b in range(B):
            l = lengths[b]
            if l > 0:
                decoder_tokens[b, :l] = decoder_cell_tokens[b]
            if max_cells > l:
                key_padding_mask[b, l:max_cells] = True
            decoder_tokens[b, max_cells : max_cells + self.n_decoder_global_tokens] = (
                decoder_global[b]
            )

        unknown_indices_tensor = torch.nn.utils.rnn.pad_sequence(
            unknown_indices,
            batch_first=True,
            padding_value=-1,
        )

        return decoder_tokens, key_padding_mask, unknown_indices_tensor

    def forward(
        self,
        x: torch.Tensor,
        full_board: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B,H,W) tensor with values 0-11
            full_board: Optional ground truth board (unused, kept for API compatibility)

        Returns:
            logits: (B,H*W) action logits
            value: (B,1) value prediction
            mine_probs: (B,H*W) mine probability predictions
        """
        del full_board  # not used in this model

        B = x.shape[0]
        device = x.device
        N = self.H * self.W

        cell_tokens = self._embed_board(x)
        flat = x.view(B, N)
        unknown_mask = flat == 10

        # Encoder: opened cells + encoder globals
        encoder_tokens, encoder_kpm = self._build_encoder_inputs(
            cell_tokens, unknown_mask
        )
        for layer in self.encoder_layers:
            encoder_tokens = layer(encoder_tokens, key_padding_mask=encoder_kpm)
        encoder_tokens = self.encoder_norm(encoder_tokens)

        # Decoder: unopened cells + decoder globals
        decoder_tokens, decoder_kpm, unknown_indices = self._build_decoder_inputs(
            cell_tokens, unknown_mask
        )
        for layer in self.decoder_layers:
            decoder_tokens = layer(
                decoder_tokens,
                encoder_tokens,
                self_key_padding_mask=decoder_kpm,
                encoder_key_padding_mask=encoder_kpm,
            )
        decoder_tokens = self.decoder_norm(decoder_tokens)

        # Reconstruct per-cell representations (B,N,d_model)
        board_tokens = cell_tokens.new_zeros(B, N, self.d_model)

        # Map encoder outputs back to known positions
        known_mask = ~unknown_mask
        for b in range(B):
            known_idx = torch.nonzero(known_mask[b], as_tuple=False).squeeze(-1)
            num_known = known_idx.shape[0]
            if num_known > 0:
                board_tokens[b, known_idx] = encoder_tokens[b, :num_known]

        # Map decoder outputs back to unknown positions
        max_dec_cells = decoder_tokens.shape[1] - self.n_decoder_global_tokens
        for b in range(B):
            idx_b = unknown_indices[b]
            valid = idx_b >= 0
            idx_valid = idx_b[valid]
            num_unknown = idx_valid.shape[0]
            if num_unknown > 0:
                board_tokens[b, idx_valid] = decoder_tokens[b, :num_unknown]

        # Global context from decoder global tokens
        dec_global_tokens = decoder_tokens[:, max_dec_cells:, :]
        global_ctx = dec_global_tokens.mean(dim=1)

        # Policy logits via pointer mechanism over all cells
        cell_keys = self.pointer_key(board_tokens)
        global_query = self.pointer_query(global_ctx).unsqueeze(1)
        attn = torch.matmul(global_query, cell_keys.transpose(1, 2)) / (
            self.d_model ** 0.5
        )
        logits = attn.squeeze(1)

        # Value head
        if self.stop_actor_grad_in_value:
            cell_mean = board_tokens.detach().mean(dim=1)
            global_ctx_detached = global_ctx.detach()
        else:
            cell_mean = board_tokens.mean(dim=1)
            global_ctx_detached = global_ctx

        value_input = torch.cat([cell_mean, global_ctx_detached], dim=1)
        value = self.value_head(value_input)

        # Mine probabilities for all cells
        mine_logits = self.mine_prob_head(board_tokens).squeeze(-1)
        mine_probs = torch.sigmoid(mine_logits)

        return logits, value, mine_probs

    def get_value(self, x: torch.Tensor, full_board: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError(
                    "action_mask must be a torch.BoolTensor shaped (B,A) after refactor"
                )
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value for the given state.

        Args:
            x: (B,H,W) tensor with values 0-11
            action: (B,) tensor with action indices (optional)
            action_mask: (B,H*W) boolean tensor indicating valid actions
            full_board: Optional ground truth board (unused)

        Returns:
            action: (B,) tensor with selected action indices
            log_prob: (B,) tensor with log probabilities of selected actions
            entropy: (B,) tensor with entropy of policy distribution
            value: (B,1) tensor with value prediction
        """
        device = x.device
        logits, value, _ = self.forward(x, full_board=full_board)

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                raise TypeError(
                    "action_mask must be a torch.BoolTensor shaped (B,A) after refactor"
                )
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()
        elif not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.long, device=device)

        return action, dist.log_prob(action), dist.entropy(), value

    def get_mine_probs(self, x: torch.Tensor) -> torch.Tensor:
        _, _, mine_probs = self.forward(x)
        return mine_probs
