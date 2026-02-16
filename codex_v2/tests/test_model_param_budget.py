from __future__ import annotations

from pathlib import Path

import yaml

from codex_v2.src.models.dual_channel_transformer_gat_v2 import DualChannelTransformerGATV2


def test_default_model_param_budget_3m() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "model_3m.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    model = DualChannelTransformerGATV2(
        weather_input_dim=int(cfg["weather_input_dim"]),
        sat_input_dim=int(cfg["sat_input_dim"]),
        weather_d_model=int(cfg["weather_d_model"]),
        sat_d_model=int(cfg["sat_d_model"]),
        weather_heads=int(cfg["weather_heads"]),
        sat_heads=int(cfg["sat_heads"]),
        weather_layers=int(cfg["weather_layers"]),
        sat_layers=int(cfg["sat_layers"]),
        weather_d_ff=int(cfg["weather_d_ff"]),
        sat_d_ff=int(cfg["sat_d_ff"]),
        dropout=float(cfg["dropout"]),
        fusion_hidden=int(cfg["fusion_hidden"]),
        gat_hidden=int(cfg["gat_hidden"]),
        gat_heads=int(cfg["gat_heads"]),
        gat_layers=int(cfg["gat_layers"]),
        max_seq_len_weather=int(cfg.get("max_seq_len_weather", 64)),
        max_seq_len_sat=int(cfg.get("max_seq_len_sat", 64)),
        opdate_embedding_dim=int(cfg.get("opdate_embedding_dim", 16)),
        opdate_vocab_size=16,
        fusion_mode=str(cfg.get("fusion_mode", "concat_gate")),
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 2_800_000 <= params <= 3_200_000, f"Expected 2.8M-3.2M params, got {params}"
