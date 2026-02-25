"""
Informer + GAT Model Definition for Wheat Yield Prediction (v2)
================================================================
Self-contained model implementing the Dual-Channel Informer encoder with
Graph Attention Network for district-level yield regression.

Architecture:
  1. Two independent InformerEncoder branches (weather + satellite)
  2. Gated feature fusion with LayerNorm
  3. Multi-layer dense GAT with residual connections + LayerNorm
  4. Improved MLP regression head with BatchNorm → scalar yield per district

Changes from v1:
  - Residual connections + LayerNorm in every GAT block
  - LayerNorm after fusion
  - Wider regression head with BatchNorm and residual path
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        return x + self.pe[:, : x.shape[1]]


# ═══════════════════════════════════════════════════════════════════════════════
# Informer Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class InformerEncoder(nn.Module):
    """
    Lightweight Informer-style sequence encoder.
    Uses Transformer encoder blocks with optional distilling (Conv1d + MaxPool)
    between layers to halve the sequence length.
    Output: masked-mean pooled embedding per input sequence.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 48,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 96,
        dropout: float = 0.2,
        distil: bool = True,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.distil = distil
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(e_layers)
            ]
        )

        if distil and e_layers > 1:
            self.distill_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                        nn.GELU(),
                        nn.MaxPool1d(kernel_size=2, stride=2),
                    )
                    for _ in range(e_layers - 1)
                ]
            )
        else:
            self.distill_layers = nn.ModuleList()

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean pooling respecting validity mask."""
        if mask is None:
            return x.mean(dim=1)
        valid = mask.to(x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            valid_mask: [batch, seq_len], 1 where valid, 0 where padded.
        Returns:
            [batch, d_model] — pooled temporal embedding.
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.dropout(h)

        mask = valid_mask
        for i, layer in enumerate(self.layers):
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = mask == 0
                all_pad = key_padding_mask.all(dim=1)
                if all_pad.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_pad, 0] = False
            h = layer(h, src_key_padding_mask=key_padding_mask)

            if self.distil and i < len(self.distill_layers):
                h = self.distill_layers[i](h.transpose(1, 2)).transpose(1, 2)
                if mask is not None:
                    m = mask.to(h.dtype).unsqueeze(1)
                    m = F.max_pool1d(m, kernel_size=2, stride=2)
                    mask = (m.squeeze(1) > 0).to(mask.dtype)

        return self._masked_mean(h, mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Attention Layer (Dense, no torch_geometric)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphAttentionLayerDense(nn.Module):
    """
    Dense multi-head graph attention layer for small fixed-size graphs
    (N=119 districts). Operates on a binary adjacency matrix without
    requiring torch_geometric.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.concat = concat

        self.lin = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

        out_features = out_dim * n_heads if concat else out_dim
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F] — node features.
            adj: [N, N] or [B, N, N] — binary adjacency (1 = connected).
        Returns:
            [B, N, out_dim*n_heads] if concat else [B, N, out_dim].
        """
        bsz, n_nodes, _ = x.shape
        h = self.lin(x).view(bsz, n_nodes, self.n_heads, self.out_dim)

        src_scores = (h * self.attn_src.view(1, 1, self.n_heads, self.out_dim)).sum(-1)
        dst_scores = (h * self.attn_dst.view(1, 1, self.n_heads, self.out_dim)).sum(-1)
        e = self.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1))

        if adj.dim() == 2:
            adj_mask = adj.unsqueeze(0).unsqueeze(-1)
        else:
            adj_mask = adj.unsqueeze(-1)
        e = e.masked_fill(adj_mask == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        out = torch.einsum("bijh,bjhc->bihc", alpha, h)
        if self.concat:
            out = out.reshape(bsz, n_nodes, self.n_heads * self.out_dim)
        else:
            out = out.mean(dim=2)
        return out + self.bias


# ═══════════════════════════════════════════════════════════════════════════════
# Residual GAT Block (NEW in v2)
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualGATBlock(nn.Module):
    """
    Wraps a GraphAttentionLayerDense with:
      - Pre-LayerNorm
      - GAT layer
      - Residual skip connection (with linear projection if dims mismatch)
      - Dropout
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gat = GraphAttentionLayerDense(
            in_dim=in_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            dropout=dropout,
            concat=concat,
        )
        gat_out_dim = out_dim * n_heads if concat else out_dim
        self.dropout = nn.Dropout(dropout)

        # Linear projection for the residual if input/output dims differ.
        if in_dim != gat_out_dim:
            self.residual_proj = nn.Linear(in_dim, gat_out_dim, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.gat(h, adj)
        h = self.dropout(h)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return h + residual


# ═══════════════════════════════════════════════════════════════════════════════
# Dual-Channel Informer + GAT (Full Model — v2)
# ═══════════════════════════════════════════════════════════════════════════════

class DualChannelInformerGAT(nn.Module):
    """
    Full architecture (v2 — improved):
      • Dual Informer encoders for weather & satellite temporal sequences
      • Gated feature fusion with LayerNorm
      • Multi-layer residual GAT over district adjacency graph
      • Improved regression head with BatchNorm → yield per node (district)

    Forward inputs:
      weather_x   [B, N, Tw, Fw]   weather features
      weather_mask[B, N, Tw]        validity mask
      sat_x       [B, N, Ts, Fs]   satellite features
      sat_mask    [B, N, Ts]        validity mask
      adj         [N, N]            adjacency matrix

    Returns:
      pred        [B, N]            yield predictions
      node_feat   [B, N, D]         intermediate node embeddings (for analysis)
    """

    def __init__(
        self,
        weather_input_dim: int,
        sat_input_dim: int,
        weather_d_model: int = 48,
        sat_d_model: int = 48,
        weather_heads: int = 4,
        sat_heads: int = 4,
        weather_layers: int = 2,
        sat_layers: int = 2,
        weather_d_ff: int = 96,
        sat_d_ff: int = 96,
        dropout: float = 0.2,
        gat_hidden: int = 48,
        gat_heads: int = 4,
        gat_layers: int = 2,
        weather_distil: bool = True,
        sat_distil: bool = True,
    ) -> None:
        super().__init__()

        # ── Temporal encoders ──────────────────────────────────────────────
        self.weather_encoder = InformerEncoder(
            input_dim=weather_input_dim,
            d_model=weather_d_model,
            n_heads=weather_heads,
            e_layers=weather_layers,
            d_ff=weather_d_ff,
            dropout=dropout,
            distil=weather_distil,
            max_len=256,
        )
        self.sat_encoder = InformerEncoder(
            input_dim=sat_input_dim,
            d_model=sat_d_model,
            n_heads=sat_heads,
            e_layers=sat_layers,
            d_ff=sat_d_ff,
            dropout=dropout,
            distil=sat_distil,
            max_len=256,
        )

        # ── Gated fusion with LayerNorm ──────────────────────────────────
        fusion_in = weather_d_model + sat_d_model
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, gat_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden, gat_hidden),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_in, gat_hidden),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(gat_hidden)  # NEW: stabilize fusion output

        # ── Residual GAT blocks (NEW: with skip connections + LayerNorm) ──
        gat_blocks = []
        in_dim = gat_hidden
        for i in range(gat_layers):
            last = i == gat_layers - 1
            block = ResidualGATBlock(
                in_dim=in_dim,
                out_dim=gat_hidden,
                n_heads=gat_heads,
                dropout=dropout,
                concat=not last,
            )
            gat_blocks.append(block)
            in_dim = gat_hidden * gat_heads if not last else gat_hidden
        self.gat_blocks = nn.ModuleList(gat_blocks)
        self.gat_out_norm = nn.LayerNorm(gat_hidden)  # final norm after GAT

        # ── Improved regression head (NEW: BatchNorm + wider) ──────────────
        self.head = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden),
            nn.BatchNorm1d(gat_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden, gat_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # lighter dropout in last layer
            nn.Linear(gat_hidden // 2, 1),
        )

    def forward(
        self,
        weather_x: torch.Tensor,
        weather_mask: torch.Tensor,
        sat_x: torch.Tensor,
        sat_mask: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_nodes = weather_x.shape[0], weather_x.shape[1]

        # Flatten batch × nodes for per-sequence encoding.
        w = weather_x.reshape(bsz * n_nodes, weather_x.shape[2], weather_x.shape[3])
        wm = weather_mask.reshape(bsz * n_nodes, weather_mask.shape[2])
        s = sat_x.reshape(bsz * n_nodes, sat_x.shape[2], sat_x.shape[3])
        sm = sat_mask.reshape(bsz * n_nodes, sat_mask.shape[2])

        w_emb = self.weather_encoder(w, wm)
        s_emb = self.sat_encoder(s, sm)

        # Gated fusion.
        fused_in = torch.cat([w_emb, s_emb], dim=-1)
        fused = self.fusion(fused_in)
        gate = self.gate(fused_in)

        # Blend weather-priority vs satellite-priority projections.
        sat_proj = F.pad(s_emb, (0, max(0, fused.shape[-1] - s_emb.shape[-1])))[
            :, : fused.shape[-1]
        ]
        weather_proj = F.pad(w_emb, (0, max(0, fused.shape[-1] - w_emb.shape[-1])))[
            :, : fused.shape[-1]
        ]
        fused = fused + gate * weather_proj + (1.0 - gate) * sat_proj
        fused = self.fusion_norm(fused)  # NEW: normalize before GAT

        # Reshape to graph layout [B, N, D] and run residual GAT.
        node_feat = fused.view(bsz, n_nodes, -1)
        h = node_feat
        for block in self.gat_blocks:
            h = block(h, adj)
        h = self.gat_out_norm(h)  # NEW: final norm

        # Regression head — needs [batch, features] for BatchNorm1d.
        h_flat = h.reshape(bsz * n_nodes, -1)
        pred_flat = self.head(h_flat).squeeze(-1)
        pred = pred_flat.view(bsz, n_nodes)

        return pred, node_feat
