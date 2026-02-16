from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from codex_v2.src.models.temporal_transformer import TemporalTransformerEncoder


class GraphAttentionLayerDense(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int,
        dropout: float,
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
        bsz, n_nodes, _ = x.shape
        h = self.lin(x).view(bsz, n_nodes, self.n_heads, self.out_dim)

        src_scores = (h * self.attn_src.view(1, 1, self.n_heads, self.out_dim)).sum(dim=-1)
        dst_scores = (h * self.attn_dst.view(1, 1, self.n_heads, self.out_dim)).sum(dim=-1)
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


class DualChannelTransformerGATV2(nn.Module):
    def __init__(
        self,
        weather_input_dim: int,
        sat_input_dim: int,
        weather_d_model: int,
        sat_d_model: int,
        weather_heads: int,
        sat_heads: int,
        weather_layers: int,
        sat_layers: int,
        weather_d_ff: int,
        sat_d_ff: int,
        dropout: float,
        fusion_hidden: int,
        gat_hidden: int,
        gat_heads: int,
        gat_layers: int,
        max_seq_len_weather: int,
        max_seq_len_sat: int,
        opdate_embedding_dim: int,
        opdate_vocab_size: int,
        fusion_mode: str = "concat_gate",
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion_mode).strip().lower()

        self.weather_encoder = TemporalTransformerEncoder(
            input_dim=weather_input_dim,
            d_model=weather_d_model,
            n_heads=weather_heads,
            n_layers=weather_layers,
            d_ff=weather_d_ff,
            dropout=dropout,
            max_len=max_seq_len_weather,
        )
        self.sat_encoder = TemporalTransformerEncoder(
            input_dim=sat_input_dim,
            d_model=sat_d_model,
            n_heads=sat_heads,
            n_layers=sat_layers,
            d_ff=sat_d_ff,
            dropout=dropout,
            max_len=max_seq_len_sat,
        )

        fusion_in = weather_d_model + sat_d_model
        if self.fusion_mode == "concat_gate":
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, fusion_hidden),
                nn.GELU(),
            )
            self.gate = nn.Sequential(
                nn.Linear(fusion_in, fusion_hidden),
                nn.Sigmoid(),
            )
            self.weather_proj = nn.Linear(weather_d_model, fusion_hidden)
            self.sat_proj = nn.Linear(sat_d_model, fusion_hidden)
            fused_dim = fusion_hidden
        elif self.fusion_mode == "cross_attention":
            self.weather_proj = nn.Linear(weather_d_model, fusion_hidden)
            self.sat_proj = nn.Linear(sat_d_model, fusion_hidden)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fusion_hidden,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_ff = nn.Sequential(
                nn.LayerNorm(fusion_hidden),
                nn.Linear(fusion_hidden, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, fusion_hidden),
            )
            fused_dim = fusion_hidden
        else:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}")

        self.opdate_embedding_dim = int(opdate_embedding_dim)
        self.opdate_embed = nn.Embedding(opdate_vocab_size, self.opdate_embedding_dim)

        self.pre_gat = nn.Sequential(
            nn.Linear(fused_dim + self.opdate_embedding_dim, gat_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden, gat_hidden),
            nn.GELU(),
        )

        blocks = []
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
            blocks.append(layer)
            in_dim = gat_hidden * gat_heads if not last else gat_hidden
        self.gat_blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden // 2, 1),
        )

    def _fuse(self, weather_emb: torch.Tensor, sat_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "concat_gate":
            fused_in = torch.cat([weather_emb, sat_emb], dim=-1)
            base = self.fusion(fused_in)
            gate = self.gate(fused_in)
            w = self.weather_proj(weather_emb)
            s = self.sat_proj(sat_emb)
            return base + gate * w + (1.0 - gate) * s

        w = self.weather_proj(weather_emb)
        s = self.sat_proj(sat_emb)
        tokens = torch.stack([w, s], dim=1)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = tokens + attn_out
        tokens = tokens + self.cross_ff(tokens)
        return tokens.mean(dim=1)

    def _expand_opdate_idx(self, bsz: int, n_nodes: int, opdate_idx: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        if opdate_idx is None:
            return torch.zeros((bsz * n_nodes,), dtype=torch.long, device=device)

        if opdate_idx.dim() == 1:
            # [B] -> [B, N]
            if opdate_idx.shape[0] != bsz:
                raise ValueError(f"opdate_idx shape mismatch: expected B={bsz}, got {opdate_idx.shape}")
            idx = opdate_idx.unsqueeze(1).repeat(1, n_nodes)
            return idx.reshape(-1).long()

        if opdate_idx.dim() == 2:
            if opdate_idx.shape[0] != bsz or opdate_idx.shape[1] != n_nodes:
                raise ValueError(
                    f"opdate_idx 2D shape mismatch: expected ({bsz}, {n_nodes}), got {tuple(opdate_idx.shape)}"
                )
            return opdate_idx.reshape(-1).long()

        raise ValueError(f"Unsupported opdate_idx dims: {opdate_idx.dim()}")

    def forward(
        self,
        weather_x: torch.Tensor,
        weather_mask: torch.Tensor,
        sat_x: torch.Tensor,
        sat_mask: torch.Tensor,
        adj: torch.Tensor,
        opdate_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # weather_x: [B, N, Tw, Fw], sat_x: [B, N, Ts, Fs]
        bsz, n_nodes = weather_x.shape[0], weather_x.shape[1]

        w = weather_x.reshape(bsz * n_nodes, weather_x.shape[2], weather_x.shape[3])
        wm = weather_mask.reshape(bsz * n_nodes, weather_mask.shape[2])
        s = sat_x.reshape(bsz * n_nodes, sat_x.shape[2], sat_x.shape[3])
        sm = sat_mask.reshape(bsz * n_nodes, sat_mask.shape[2])

        w_emb = self.weather_encoder(w, wm)
        s_emb = self.sat_encoder(s, sm)
        fused = self._fuse(w_emb, s_emb)

        op_idx = self._expand_opdate_idx(bsz, n_nodes, opdate_idx, device=fused.device)
        op_emb = self.opdate_embed(op_idx)
        fused = torch.cat([fused, op_emb], dim=-1)
        fused = self.pre_gat(fused)

        node_feat = fused.view(bsz, n_nodes, -1)
        h = node_feat
        for i, gat in enumerate(self.gat_blocks):
            h = gat(h, adj)
            if i < len(self.gat_blocks) - 1:
                h = F.elu(h)

        pred = self.head(h).squeeze(-1)
        return pred, node_feat
