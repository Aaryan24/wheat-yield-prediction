from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from codex_v2.src.models.temporal_transformer import MaskedAttentionPooling, TemporalTransformerEncoder


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


@dataclass
class ModelForwardOutput:
    pred: torch.Tensor
    node_feat: torch.Tensor
    pred_raw_delta: Optional[torch.Tensor] = None
    class_logits: Optional[torch.Tensor] = None
    class_probs: Optional[torch.Tensor] = None
    drop_mag: Optional[torch.Tensor] = None
    rise_mag: Optional[torch.Tensor] = None
    drop_logmag: Optional[torch.Tensor] = None
    rise_logmag: Optional[torch.Tensor] = None


class BidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.weather_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.sat_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.weather_norm1 = nn.LayerNorm(embed_dim)
        self.sat_norm1 = nn.LayerNorm(embed_dim)
        self.weather_norm2 = nn.LayerNorm(embed_dim)
        self.sat_norm2 = nn.LayerNorm(embed_dim)
        self.weather_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.sat_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _key_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        key_padding_mask = mask <= 0
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
        return key_padding_mask

    def forward(
        self,
        weather_x: torch.Tensor,
        weather_mask: Optional[torch.Tensor],
        sat_x: torch.Tensor,
        sat_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sat_kpm = self._key_padding_mask(sat_mask)
        weather_kpm = self._key_padding_mask(weather_mask)

        w_attn, _ = self.weather_attn(
            query=weather_x,
            key=sat_x,
            value=sat_x,
            key_padding_mask=sat_kpm,
            need_weights=False,
        )
        s_attn, _ = self.sat_attn(
            query=sat_x,
            key=weather_x,
            value=weather_x,
            key_padding_mask=weather_kpm,
            need_weights=False,
        )

        weather_h = self.weather_norm1(weather_x + self.dropout(w_attn))
        sat_h = self.sat_norm1(sat_x + self.dropout(s_attn))
        weather_h = self.weather_norm2(weather_h + self.dropout(self.weather_ff(weather_h)))
        sat_h = self.sat_norm2(sat_h + self.dropout(self.sat_ff(sat_h)))
        return weather_h, sat_h


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
        pooling_mode_weather: str = "mean",
        pooling_mode_sat: str = "mean",
        cross_attention_level: str = "pooled",
        cross_attention_heads: int = 4,
        head_mode: str = "scalar",
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion_mode).strip().lower()
        self.pooling_mode_weather = str(pooling_mode_weather).strip().lower()
        self.pooling_mode_sat = str(pooling_mode_sat).strip().lower()
        self.cross_attention_level = str(cross_attention_level).strip().lower()
        self.head_mode = str(head_mode).strip().lower()

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
        self.weather_pool = MaskedAttentionPooling(d_model=weather_d_model, dropout=dropout)
        self.sat_pool = MaskedAttentionPooling(d_model=sat_d_model, dropout=dropout)

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
            if self.cross_attention_level == "sequence":
                self.cross_block = BidirectionalCrossAttentionBlock(
                    embed_dim=fusion_hidden,
                    num_heads=int(cross_attention_heads),
                    dropout=dropout,
                )
                self.weather_fused_pool = MaskedAttentionPooling(d_model=fusion_hidden, dropout=dropout)
                self.sat_fused_pool = MaskedAttentionPooling(d_model=fusion_hidden, dropout=dropout)
                self.seq_fusion = nn.Sequential(
                    nn.Linear(fusion_hidden * 2, fusion_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_hidden, fusion_hidden),
                    nn.GELU(),
                )
            else:
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=fusion_hidden,
                    num_heads=int(cross_attention_heads),
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

        if self.head_mode == "sign_magnitude":
            hidden = max(gat_hidden // 2, 32)
            self.classifier_head = nn.Sequential(
                nn.Linear(gat_hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 3),
            )
            self.drop_head = nn.Sequential(
                nn.Linear(gat_hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.rise_head = nn.Sequential(
                nn.Linear(gat_hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(gat_hidden, gat_hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gat_hidden // 2, 1),
            )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        valid = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def _pool_branch(self, seq: torch.Tensor, mask: Optional[torch.Tensor], branch: str) -> torch.Tensor:
        mode = self.pooling_mode_weather if branch == "weather" else self.pooling_mode_sat
        if mode == "attention":
            pool = self.weather_pool if branch == "weather" else self.sat_pool
            pooled, _ = pool(seq, mask)
            return pooled
        return self._masked_mean(seq, mask)

    def _fuse(
        self,
        weather_seq: torch.Tensor,
        weather_mask: Optional[torch.Tensor],
        sat_seq: torch.Tensor,
        sat_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.fusion_mode == "concat_gate":
            weather_emb = self._pool_branch(weather_seq, weather_mask, branch="weather")
            sat_emb = self._pool_branch(sat_seq, sat_mask, branch="sat")
            fused_in = torch.cat([weather_emb, sat_emb], dim=-1)
            base = self.fusion(fused_in)
            gate = self.gate(fused_in)
            w = self.weather_proj(weather_emb)
            s = self.sat_proj(sat_emb)
            return base + gate * w + (1.0 - gate) * s

        if self.cross_attention_level == "sequence":
            w_seq = self.weather_proj(weather_seq)
            s_seq = self.sat_proj(sat_seq)
            w_seq, s_seq = self.cross_block(
                weather_x=w_seq,
                weather_mask=weather_mask,
                sat_x=s_seq,
                sat_mask=sat_mask,
            )
            w_pooled, _ = self.weather_fused_pool(w_seq, weather_mask)
            s_pooled, _ = self.sat_fused_pool(s_seq, sat_mask)
            return self.seq_fusion(torch.cat([w_pooled, s_pooled], dim=-1))

        weather_emb = self._pool_branch(weather_seq, weather_mask, branch="weather")
        sat_emb = self._pool_branch(sat_seq, sat_mask, branch="sat")
        w = self.weather_proj(weather_emb)
        s = self.sat_proj(sat_emb)
        tokens = torch.stack([w, s], dim=1)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens, need_weights=False)
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
    ) -> ModelForwardOutput:
        # weather_x: [B, N, Tw, Fw], sat_x: [B, N, Ts, Fs]
        bsz, n_nodes = weather_x.shape[0], weather_x.shape[1]

        w = weather_x.reshape(bsz * n_nodes, weather_x.shape[2], weather_x.shape[3])
        wm = weather_mask.reshape(bsz * n_nodes, weather_mask.shape[2])
        s = sat_x.reshape(bsz * n_nodes, sat_x.shape[2], sat_x.shape[3])
        sm = sat_mask.reshape(bsz * n_nodes, sat_mask.shape[2])

        w_seq = self.weather_encoder(w, wm, return_sequence=True)
        s_seq = self.sat_encoder(s, sm, return_sequence=True)
        fused = self._fuse(w_seq, wm, s_seq, sm)

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

        if self.head_mode == "sign_magnitude":
            logits = self.classifier_head(h)
            probs = torch.softmax(logits, dim=-1)
            drop_logmag = F.softplus(self.drop_head(h).squeeze(-1))
            rise_logmag = F.softplus(self.rise_head(h).squeeze(-1))
            drop_mag = torch.expm1(drop_logmag)
            rise_mag = torch.expm1(rise_logmag)
            pred_delta = (probs[..., 2] * rise_mag) - (probs[..., 0] * drop_mag)
            return ModelForwardOutput(
                pred=pred_delta,
                node_feat=node_feat,
                pred_raw_delta=pred_delta,
                class_logits=logits,
                class_probs=probs,
                drop_mag=drop_mag,
                rise_mag=rise_mag,
                drop_logmag=drop_logmag,
                rise_logmag=rise_logmag,
            )

        pred = self.head(h).squeeze(-1)
        return ModelForwardOutput(pred=pred, node_feat=node_feat)
