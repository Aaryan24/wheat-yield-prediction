from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.shape[1]
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len={self.max_len}")
        return x + self.pos[:, :t, :]


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] (1 valid, 0 missing)
        if mask is None:
            return x.mean(dim=1)
        valid = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.positional(h)
        h = self.norm(h)
        h = self.dropout(h)

        key_padding_mask = None
        mask = valid_mask
        if mask is not None:
            key_padding_mask = mask == 0
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self._masked_mean(h, valid_mask)
