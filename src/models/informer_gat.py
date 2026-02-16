from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


class InformerEncoder(nn.Module):
    """
    Lightweight Informer-style sequence encoder.
    Uses Transformer encoder blocks plus optional distilling between layers.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
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
        # x: [batch, seq, dim], mask: [batch, seq] with 1 for valid.
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
        # x: [batch, seq_len, input_dim]
        # valid_mask: [batch, seq_len], 1 where valid.
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.dropout(h)

        mask = valid_mask
        for i, layer in enumerate(self.layers):
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = mask == 0
                # Avoid all-masked rows, which can produce NaNs in attention.
                all_pad = key_padding_mask.all(dim=1)
                if all_pad.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_pad, 0] = False
            h = layer(h, src_key_padding_mask=key_padding_mask)

            if self.distil and i < len(self.distill_layers):
                # [B, T, C] -> [B, C, T] -> distill -> [B, T2, C]
                h = self.distill_layers[i](h.transpose(1, 2)).transpose(1, 2)
                if mask is not None:
                    # Downsample valid mask conservatively: any valid element in each pool window.
                    m = mask.to(h.dtype).unsqueeze(1)
                    m = F.max_pool1d(m, kernel_size=2, stride=2)
                    mask = (m.squeeze(1) > 0).to(mask.dtype)

        return self._masked_mean(h, mask)


class GraphAttentionLayerDense(nn.Module):
    """
    Dense GAT layer for small fixed-size graphs (N=119 here), no torch_geometric dependency.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
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
        # x: [B, N, F]
        # adj: [N, N] or [B, N, N], 1 for connected.
        bsz, n_nodes, _ = x.shape
        h = self.lin(x).view(bsz, n_nodes, self.n_heads, self.out_dim)  # [B, N, H, C]

        src_scores = (h * self.attn_src.view(1, 1, self.n_heads, self.out_dim)).sum(dim=-1)  # [B, N, H]
        dst_scores = (h * self.attn_dst.view(1, 1, self.n_heads, self.out_dim)).sum(dim=-1)  # [B, N, H]
        e = self.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1))  # [B, N, N, H]

        if adj.dim() == 2:
            adj_mask = adj.unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        else:
            adj_mask = adj.unsqueeze(-1)  # [B, N, N, 1]
        e = e.masked_fill(adj_mask == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        out = torch.einsum("bijh,bjhc->bihc", alpha, h)  # [B, N, H, C]
        if self.concat:
            out = out.reshape(bsz, n_nodes, self.n_heads * self.out_dim)
        else:
            out = out.mean(dim=2)
        out = out + self.bias
        return out


class DualChannelInformerGAT(nn.Module):
    def __init__(
        self,
        weather_input_dim: int,
        sat_input_dim: int,
        weather_d_model: int = 64,
        sat_d_model: int = 64,
        weather_heads: int = 4,
        sat_heads: int = 4,
        weather_layers: int = 2,
        sat_layers: int = 2,
        weather_d_ff: int = 128,
        sat_d_ff: int = 128,
        dropout: float = 0.1,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        weather_distil: bool = True,
        sat_distil: bool = True,
    ) -> None:
        super().__init__()
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

        gat_blocks = []
        in_dim = gat_hidden
        for i in range(gat_layers):
            last = i == gat_layers - 1
            layer = GraphAttentionLayerDense(
                in_dim=in_dim,
                out_dim=gat_hidden,
                n_heads=gat_heads,
                dropout=dropout,
                concat=not last,
            )
            gat_blocks.append(layer)
            in_dim = gat_hidden * gat_heads if not last else gat_hidden
        self.gat_blocks = nn.ModuleList(gat_blocks)

        self.head = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
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
        # weather_x: [B, N, Tw, Fw], sat_x: [B, N, Ts, Fs]
        bsz, n_nodes = weather_x.shape[0], weather_x.shape[1]
        w = weather_x.reshape(bsz * n_nodes, weather_x.shape[2], weather_x.shape[3])
        wm = weather_mask.reshape(bsz * n_nodes, weather_mask.shape[2])
        s = sat_x.reshape(bsz * n_nodes, sat_x.shape[2], sat_x.shape[3])
        sm = sat_mask.reshape(bsz * n_nodes, sat_mask.shape[2])

        w_emb = self.weather_encoder(w, wm)
        s_emb = self.sat_encoder(s, sm)

        fused_in = torch.cat([w_emb, s_emb], dim=-1)
        fused = self.fusion(fused_in)
        gate = self.gate(fused_in)
        # Blend weather and satellite information; weather gets default priority when satellite is weak.
        sat_proj = F.pad(s_emb, (0, max(0, fused.shape[-1] - s_emb.shape[-1])))[:, : fused.shape[-1]]
        weather_proj = F.pad(w_emb, (0, max(0, fused.shape[-1] - w_emb.shape[-1])))[:, : fused.shape[-1]]
        fused = fused + gate * weather_proj + (1.0 - gate) * sat_proj

        node_feat = fused.view(bsz, n_nodes, -1)
        h = node_feat
        for i, gat in enumerate(self.gat_blocks):
            h = gat(h, adj)
            if i < len(self.gat_blocks) - 1:
                h = F.elu(h)

        pred = self.head(h).squeeze(-1)  # [B, N]
        return pred, node_feat
