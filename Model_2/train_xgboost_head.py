#!/usr/bin/env python3
"""
train_xgboost_head.py — Train Informer + GAT Backbone → XGBoost Head
=====================================================================
Two-stage pipeline:
  Stage 1: Train the full DualChannelInformerGAT model (with MLP head) to learn
           spatial-temporal embeddings via the Informer encoders and GAT.
  Stage 2: Freeze the trained backbone, extract post-GAT node embeddings, and
           train an XGBRegressor on those embeddings for final yield prediction.

Usage:
    python Model_2/train_xgboost_head.py                     # full run
    python Model_2/train_xgboost_head.py --epochs 3          # quick smoke test
    python Model_2/train_xgboost_head.py --config Model_2/config.yaml

Analysis artifacts (same 7 as train.py) are auto-saved to Model_2/analysis/run_X/.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server / headless use.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError(
        "xgboost is required for this script. Install it via:\n"
        "  pip install xgboost"
    )

# Import the model from the same folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from informer_gat_model import DualChannelInformerGAT  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(f"[XGB-Head] {msg}", flush=True)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _to(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _next_run_dir(analysis_dir: Path) -> Path:
    """Find the next available run_X folder inside analysis_dir."""
    analysis_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in analysis_dir.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            try:
                existing.append(int(p.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    next_id = max(existing, default=0) + 1
    run_dir = analysis_dir / f"run_{next_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Splitting
# ═══════════════════════════════════════════════════════════════════════════════

def _season_split(
    years: Sequence[int], mode: str, seed: int,
    custom_train: Optional[List[int]] = None,
    custom_val: Optional[List[int]] = None,
    custom_test: Optional[List[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Split seasons into train / val / test."""
    if mode == "custom":
        if custom_train is None or custom_val is None or custom_test is None:
            raise ValueError("custom mode needs train_years, val_years, test_years")
        return list(custom_train), list(custom_val), list(custom_test)
    if mode == "fixed":
        sorted_y = sorted(years)
        return sorted_y[:-2], [sorted_y[-2]], [sorted_y[-1]]
    if mode == "random":
        import random as _rng
        _rng.seed(seed)
        shuffled = list(years)
        _rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, n // 5)
        n_val = max(1, (n - n_test) // 5)
        return shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test]
    raise ValueError(f"Unknown split mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_scaler(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std from valid entries. values: [S,N,T,F], mask: [S,N,T]."""
    expanded = mask[..., np.newaxis].astype(bool)
    n_feats = values.shape[-1]
    mean = np.zeros(n_feats, dtype=np.float64)
    std = np.ones(n_feats, dtype=np.float64)
    for f in range(n_feats):
        valid = values[..., f][expanded[..., 0]]
        if len(valid) > 0:
            mean[f] = valid.mean()
            std[f] = valid.std()
            if std[f] < 1e-8:
                std[f] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    safe = np.where(np.abs(y_true) > 1e-6, y_true, 1e-6)
    mape = float(np.mean(np.abs(diff / safe)) * 100.0)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis generators
# ═══════════════════════════════════════════════════════════════════════════════

def _save_metrics(
    train_m: dict, val_m: dict, test_m: dict, out_dir: Path
) -> None:
    data = {"train": train_m, "val": val_m, "test": test_m}
    path = out_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(f"  → {path}")


def _save_predictions(
    district_ids: np.ndarray,
    district_names: np.ndarray,
    state_names: np.ndarray,
    season_years: np.ndarray,
    split_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """Save full district-level predictions CSV."""
    import pandas as pd

    rows = []
    for si, year in enumerate(season_years):
        for di in range(len(district_ids)):
            actual = float(y_true[si, di])
            predicted = float(y_pred[si, di])
            error = predicted - actual
            abs_err = abs(error)
            per_mape = (abs_err / abs(actual) * 100.0) if abs(actual) > 1e-6 else float("nan")
            rows.append({
                "split": str(split_labels[si]),
                "season_year": int(year),
                "district_id": str(district_ids[di]),
                "district_name": str(district_names[di]),
                "state_name": str(state_names[di]),
                "actual_yield_kg_per_ha": actual,
                "predicted_yield_kg_per_ha": predicted,
                "error_kg_per_ha": error,
                "abs_error_kg_per_ha": abs_err,
                "mape_percent": per_mape,
            })
    df = pd.DataFrame(rows)
    path = out_dir / "predictions.csv"
    df.to_csv(path, index=False)
    _log(f"  → {path}")


def _save_training_curves(
    train_losses: List[float], val_losses: List[float], out_dir: Path
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="Train Loss (Backbone)", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss (Backbone)", color="#F44336", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Backbone Training Curves (Informer + GAT)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  → {path}")


def _save_error_analysis(
    y_true_all: np.ndarray, y_pred_all: np.ndarray, out_dir: Path
) -> None:
    errors = y_pred_all - y_true_all

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: actual vs predicted.
    ax = axes[0]
    ax.scatter(y_true_all, y_pred_all, alpha=0.4, s=15, c="#1976D2", edgecolors="none")
    lo = min(y_true_all.min(), y_pred_all.min()) * 0.9
    hi = max(y_true_all.max(), y_pred_all.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual Yield (kg/ha)", fontsize=11)
    ax.set_ylabel("Predicted Yield (kg/ha)", fontsize=11)
    ax.set_title("Actual vs Predicted (XGBoost Head)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Histogram: errors.
    ax = axes[1]
    ax.hist(errors, bins=40, color="#66BB6A", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Error (kg/ha)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Distribution (XGBoost Head)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "error_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  → {path}")


def _save_model_params(
    model: nn.Module, model_kwargs: dict, cfg: dict,
    xgb_params: dict, out_dir: Path
) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-layer breakdown.
    layer_info = []
    for name, param in model.named_parameters():
        layer_info.append({
            "name": name,
            "shape": list(param.shape),
            "params": int(param.numel()),
            "trainable": bool(param.requires_grad),
        })

    data = {
        "architecture": "DualChannelInformerGAT (backbone) + XGBoost (head)",
        "backbone_total_params": int(total),
        "backbone_trainable_params": int(trainable),
        "backbone_hyperparameters": model_kwargs,
        "xgboost_hyperparameters": xgb_params,
        "training_config": {
            "backbone_epochs": cfg["training"]["epochs"],
            "backbone_lr": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "gradient_clip": cfg["training"]["gradient_clip"],
            "seed": cfg["training"]["seed"],
        },
        "layer_breakdown": layer_info,
    }

    path = out_dir / "model_params.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Accuracy Classification (MAPE-based)
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_accuracy(mape_pct: float) -> str:
    """Classify a single prediction by its MAPE percentage."""
    if np.isnan(mape_pct):
        return "unknown"
    if mape_pct < 2.0:
        return "accurate"
    elif mape_pct < 5.0:
        return "somewhat_accurate"
    elif mape_pct < 10.0:
        return "somewhat_inaccurate"
    else:
        return "inaccurate"


def _save_accuracy_chart(
    df: "pd.DataFrame", title_suffix: str, out_dir: Path, file_prefix: str,
) -> None:
    """Save accuracy classification bar chart for one subset of predictions."""
    cat_order = ["accurate", "somewhat_accurate", "somewhat_inaccurate", "inaccurate"]
    cat_labels = ["Accurate\n(MAPE<2%)", "Somewhat\nAccurate\n(2-5%)",
                  "Somewhat\nInaccurate\n(5-10%)", "Inaccurate\n(>10%)"]
    cat_colors = ["#4CAF50", "#8BC34A", "#FF9800", "#F44336"]

    total = len(df)
    counts = [int((df["accuracy_category"] == c).sum()) for c in cat_order]
    pcts = [c / max(total, 1) * 100 for c in counts]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(cat_labels, counts, color=cat_colors, edgecolor="white", linewidth=1.2)
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Predictions", fontsize=11)
    ax.set_title(f"Accuracy Classification{title_suffix}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    png_path = out_dir / f"{file_prefix}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  → {png_path}")


def _save_accuracy_classification(
    district_ids: np.ndarray,
    district_names: np.ndarray,
    state_names: np.ndarray,
    season_years: np.ndarray,
    split_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Classify each prediction into accuracy buckets based on per-prediction MAPE:
      - Accurate         : MAPE < 2%
      - Somewhat Accurate: MAPE 2–5%
      - Somewhat Inaccurate: MAPE 5–10%
      - Inaccurate       : MAPE > 10%
    Saves combined + per-split CSVs and charts.
    """
    import pandas as pd

    rows = []
    for si, year in enumerate(season_years):
        for di in range(len(district_ids)):
            actual = float(y_true[si, di])
            predicted = float(y_pred[si, di])
            abs_err = abs(predicted - actual)
            mape_pct = (abs_err / abs(actual) * 100.0) if abs(actual) > 1e-6 else float("nan")
            category = _classify_accuracy(mape_pct)
            rows.append({
                "split": str(split_labels[si]),
                "season_year": int(year),
                "district_id": str(district_ids[di]),
                "district_name": str(district_names[di]),
                "state_name": str(state_names[di]),
                "actual_yield": actual,
                "predicted_yield": predicted,
                "mape_percent": mape_pct,
                "accuracy_category": category,
            })

    df = pd.DataFrame(rows)

    # ── Combined (all splits) ────────────────────────────────────────────
    csv_path = out_dir / "accuracy_classification_all.csv"
    df.to_csv(csv_path, index=False)
    _log(f"  → {csv_path}")
    _save_accuracy_chart(df, " (All)", out_dir, "accuracy_classification_all")

    # ── Per-split outputs ────────────────────────────────────────────────
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        if len(split_df) == 0:
            continue

        split_csv = out_dir / f"accuracy_classification_{split_name}.csv"
        split_df.to_csv(split_csv, index=False)
        _log(f"  → {split_csv}")
        _save_accuracy_chart(
            split_df, f" ({split_name.title()})",
            out_dir, f"accuracy_classification_{split_name}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# District-wise Trend Analysis (direction match + precision/recall)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_trend_report(df: "pd.DataFrame") -> dict:
    """Compute precision / recall / F1 per trend direction from a trend DataFrame."""
    directions = ["increase", "decrease", "stable"]
    report = {}
    overall_correct = 0
    overall_total = len(df)

    for d in directions:
        tp = int(((df["actual_direction"] == d) & (df["predicted_direction"] == d)).sum())
        fp = int(((df["actual_direction"] != d) & (df["predicted_direction"] == d)).sum())
        fn = int(((df["actual_direction"] == d) & (df["predicted_direction"] != d)).sum())
        tn = int(((df["actual_direction"] != d) & (df["predicted_direction"] != d)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        report[d] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }
        overall_correct += tp

    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    macro_prec = np.mean([report[d]["precision"] for d in directions])
    macro_rec = np.mean([report[d]["recall"] for d in directions])
    macro_f1 = np.mean([report[d]["f1_score"] for d in directions])

    total_support = sum(report[d]["support"] for d in directions)
    if total_support > 0:
        w_prec = sum(report[d]["precision"] * report[d]["support"] for d in directions) / total_support
        w_rec = sum(report[d]["recall"] * report[d]["support"] for d in directions) / total_support
        w_f1 = sum(report[d]["f1_score"] * report[d]["support"] for d in directions) / total_support
    else:
        w_prec = w_rec = w_f1 = 0.0

    return {
        "per_class": report,
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_direction_matches": overall_correct,
        "overall_total_transitions": overall_total,
        "macro_avg": {
            "precision": round(float(macro_prec), 4),
            "recall": round(float(macro_rec), 4),
            "f1_score": round(float(macro_f1), 4),
        },
        "weighted_avg": {
            "precision": round(float(w_prec), 4),
            "recall": round(float(w_rec), 4),
            "f1_score": round(float(w_f1), 4),
        },
    }


def _save_trend_visualization(
    df: "pd.DataFrame", report: dict, title_suffix: str, out_dir: Path, file_prefix: str,
) -> None:
    """Save confusion matrix + precision/recall/F1 bar chart for one trend subset."""
    directions = ["increase", "decrease", "stable"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion-style heatmap.
    ax = axes[0]
    conf = np.zeros((3, 3), dtype=int)
    for i, ad in enumerate(directions):
        for j, pd_ in enumerate(directions):
            conf[i, j] = int(((df["actual_direction"] == ad) & (df["predicted_direction"] == pd_)).sum())
    im = ax.imshow(conf, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([d.title() for d in directions], fontsize=10)
    ax.set_yticklabels([d.title() for d in directions], fontsize=10)
    ax.set_xlabel("Predicted Direction", fontsize=11)
    ax.set_ylabel("Actual Direction", fontsize=11)
    ax.set_title(f"Trend Confusion Matrix{title_suffix}", fontsize=13, fontweight="bold")
    for i in range(3):
        for j in range(3):
            color = "white" if conf[i, j] > conf.max() * 0.5 else "black"
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Precision / Recall / F1 grouped bars.
    ax = axes[1]
    per_class = report["per_class"]
    x_pos = np.arange(len(directions))
    width = 0.25
    precs = [per_class[d]["precision"] for d in directions]
    recs = [per_class[d]["recall"] for d in directions]
    f1s = [per_class[d]["f1_score"] for d in directions]
    ax.bar(x_pos - width, precs, width, label="Precision", color="#2196F3")
    ax.bar(x_pos, recs, width, label="Recall", color="#4CAF50")
    ax.bar(x_pos + width, f1s, width, label="F1 Score", color="#FF9800")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.title() for d in directions], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Trend Classification Metrics{title_suffix}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    png_path = out_dir / f"{file_prefix}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  → {png_path}")


def _save_trend_analysis(
    district_ids: np.ndarray,
    district_names: np.ndarray,
    state_names: np.ndarray,
    season_years: np.ndarray,
    split_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """
    For consecutive season pairs, compare whether the predicted trend direction
    (increase / decrease / stable) matches the actual trend direction per district.

    Generates SEPARATE outputs for each split (train, val, test) plus an overall
    combined report:
      - trend_analysis_all.csv / .json / .png
      - trend_analysis_train.csv / .json / .png
      - trend_analysis_val.csv / .json / .png
      - trend_analysis_test.csv / .json / .png
    """
    import pandas as pd

    sorted_idx = np.argsort(season_years)
    sorted_years = season_years[sorted_idx]
    sorted_true = y_true[sorted_idx]    # [S, N]
    sorted_pred = y_pred[sorted_idx]
    sorted_splits = split_labels[sorted_idx]

    # Build year → split mapping.
    year_to_split = {int(sorted_years[i]): sorted_splits[i] for i in range(len(sorted_years))}

    THRESHOLD_PCT = 0.5  # ±0.5% change => "stable"

    rows = []
    for ti in range(len(sorted_years) - 1):
        year_from = int(sorted_years[ti])
        year_to = int(sorted_years[ti + 1])
        # Tag transition by the split of the target (year_to) season.
        transition_split = year_to_split.get(year_to, "unknown")

        for di in range(len(district_ids)):
            actual_from = sorted_true[ti, di]
            actual_to = sorted_true[ti + 1, di]
            pred_from = sorted_pred[ti, di]
            pred_to = sorted_pred[ti + 1, di]

            # Actual direction.
            actual_change_pct = ((actual_to - actual_from) / max(abs(actual_from), 1e-6)) * 100
            if actual_change_pct > THRESHOLD_PCT:
                actual_dir = "increase"
            elif actual_change_pct < -THRESHOLD_PCT:
                actual_dir = "decrease"
            else:
                actual_dir = "stable"

            # Predicted direction.
            pred_change_pct = ((pred_to - pred_from) / max(abs(pred_from), 1e-6)) * 100
            if pred_change_pct > THRESHOLD_PCT:
                pred_dir = "increase"
            elif pred_change_pct < -THRESHOLD_PCT:
                pred_dir = "decrease"
            else:
                pred_dir = "stable"

            direction_match = actual_dir == pred_dir

            rows.append({
                "split": transition_split,
                "district_id": str(district_ids[di]),
                "district_name": str(district_names[di]),
                "state_name": str(state_names[di]),
                "year_from": year_from,
                "year_to": year_to,
                "actual_yield_from": float(actual_from),
                "actual_yield_to": float(actual_to),
                "actual_change_pct": float(actual_change_pct),
                "actual_direction": actual_dir,
                "predicted_yield_from": float(pred_from),
                "predicted_yield_to": float(pred_to),
                "predicted_change_pct": float(pred_change_pct),
                "predicted_direction": pred_dir,
                "direction_match": bool(direction_match),
            })

    df = pd.DataFrame(rows)

    # ── Save combined (all splits) ───────────────────────────────────────
    csv_path = out_dir / "trend_analysis_all.csv"
    df.to_csv(csv_path, index=False)
    _log(f"  → {csv_path}")

    all_report = _compute_trend_report(df)
    report_path = out_dir / "trend_analysis_all.json"
    with open(report_path, "w") as f:
        json.dump(all_report, f, indent=2)
    _log(f"  → {report_path}")
    _save_trend_visualization(df, all_report, " (All)", out_dir, "trend_analysis_all")

    # ── Save per-split outputs ───────────────────────────────────────────
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        if len(split_df) == 0:
            _log(f"  ⚠ No trend transitions for split '{split_name}' (need ≥2 consecutive years), skipping.")
            continue

        # CSV
        split_csv = out_dir / f"trend_analysis_{split_name}.csv"
        split_df.to_csv(split_csv, index=False)
        _log(f"  → {split_csv}")

        # Classification report JSON
        split_report = _compute_trend_report(split_df)
        split_json = out_dir / f"trend_analysis_{split_name}.json"
        with open(split_json, "w") as f:
            json.dump(split_report, f, indent=2)
        _log(f"  → {split_json}")

        # Visualization PNG
        _save_trend_visualization(
            split_df, split_report,
            f" ({split_name.title()})",
            out_dir, f"trend_analysis_{split_name}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Two-Stage Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Informer + GAT backbone → XGBoost head, auto-generate analysis."
    )
    parser.add_argument(
        "--config", type=str, default="Model_2/config.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset path (default: from config).",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override backbone training epochs.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override backbone learning rate.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (auto/cuda/cpu/mps).")
    # XGBoost-specific arguments.
    parser.add_argument("--xgb-n-estimators", type=int, default=500,
                        help="XGBoost number of boosting rounds.")
    parser.add_argument("--xgb-max-depth", type=int, default=6,
                        help="XGBoost max tree depth.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05,
                        help="XGBoost learning rate (eta).")
    parser.add_argument("--xgb-subsample", type=float, default=0.8,
                        help="XGBoost row subsample ratio.")
    parser.add_argument("--xgb-colsample", type=float, default=0.8,
                        help="XGBoost column subsample ratio per tree.")
    parser.add_argument("--xgb-patience", type=int, default=50,
                        help="XGBoost early stopping rounds.")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    cfg = _load_yaml(Path(args.config))
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    split_cfg = cfg["split"]

    epochs = args.epochs or train_cfg["epochs"]
    lr = args.lr or train_cfg["lr"]
    weight_decay = train_cfg["weight_decay"]
    grad_clip = train_cfg["gradient_clip"]
    seed = train_cfg["seed"]
    device_name = args.device or train_cfg["device"]
    scheduler_type = train_cfg.get("scheduler", "none")
    loss_type = train_cfg.get("loss", "mse")
    normalize_yield = train_cfg.get("normalize_yield", False)

    dataset_path = Path(args.dataset or data_cfg["dataset_file"])
    analysis_root = Path(cfg["output"]["analysis_dir"])

    # ── Create auto-incrementing run folder ──────────────────────────────
    run_dir = _next_run_dir(analysis_root)
    _log(f"Run output directory: {run_dir}")

    _set_seed(seed)
    device = _pick_device(device_name)
    _log(f"Device: {device}")
    _log(f"Seed: {seed}")

    # ── Load dataset ─────────────────────────────────────────────────────
    _log(f"Loading dataset from {dataset_path}...")
    ds = np.load(dataset_path, allow_pickle=True)
    weather_x = ds["weather_x"]          # [S, N, Tw, Fw]
    weather_mask = ds["weather_mask"]     # [S, N, Tw]
    sat_x = ds["sat_x"]                  # [S, N, Ts, Fs]
    sat_mask = ds["sat_mask"]            # [S, N, Ts]
    yields = ds["yields"]                # [S, N]
    adj_np = ds["adjacency"]            # [N, N]
    district_ids = ds["district_ids"]
    season_years = ds["season_years"]
    district_names = ds["district_names"]
    state_names = ds["state_names"]

    n_weather_feats = weather_x.shape[-1]
    n_sat_feats = sat_x.shape[-1]
    n_districts = weather_x.shape[1]
    n_seasons = weather_x.shape[0]

    _log(f"  Seasons: {list(season_years)} ({n_seasons})")
    _log(f"  Districts: {n_districts}")
    _log(f"  Weather shape: {weather_x.shape}")
    _log(f"  Satellite shape: {sat_x.shape}")

    # ── Exclude states entirely ──────────────────────────────────────────
    exclude_states = split_cfg.get("exclude_states", [])
    if exclude_states:
        keep_mask = np.array([s not in exclude_states for s in state_names])
        keep_idx = np.where(keep_mask)[0]
        n_removed = n_districts - len(keep_idx)
        _log(f"  Excluding states {exclude_states}: removing {n_removed} districts, keeping {len(keep_idx)}")

        weather_x = weather_x[:, keep_idx]
        weather_mask = weather_mask[:, keep_idx]
        sat_x = sat_x[:, keep_idx]
        sat_mask = sat_mask[:, keep_idx]
        yields = yields[:, keep_idx]
        adj_np = adj_np[np.ix_(keep_idx, keep_idx)]
        district_ids = district_ids[keep_idx]
        district_names = district_names[keep_idx]
        state_names = state_names[keep_idx]

        n_districts = len(keep_idx)
        _log(f"  After filtering: {n_districts} districts, weather={weather_x.shape}, sat={sat_x.shape}")

    # ── Split ────────────────────────────────────────────────────────────
    train_years, val_years, test_years = _season_split(
        list(season_years), split_cfg["mode"], split_cfg["seed"],
        custom_train=split_cfg.get("train_years"),
        custom_val=split_cfg.get("val_years"),
        custom_test=split_cfg.get("test_years"),
    )
    year_to_idx = {int(y): i for i, y in enumerate(season_years)}
    idx_train = np.array([year_to_idx[y] for y in train_years], dtype=int)
    idx_val = np.array([year_to_idx[y] for y in val_years], dtype=int)
    idx_test = np.array([year_to_idx[y] for y in test_years], dtype=int)
    _log(f"  Train: {train_years}, Val: {val_years}, Test: {test_years}")

    # ── Scale features (fit on train only) ─────────────────────────────
    w_mean, w_std = _fit_scaler(weather_x[idx_train], weather_mask[idx_train])
    s_mean, s_std = _fit_scaler(sat_x[idx_train], sat_mask[idx_train])
    weather_x = _apply_scaler(weather_x, w_mean, w_std)
    sat_x = _apply_scaler(sat_x, s_mean, s_std)

    # ── Yield normalization (z-score, fit on train) ───────────────────
    if normalize_yield:
        y_train_vals = yields[idx_train].ravel()
        y_mean = float(np.mean(y_train_vals))
        y_std = float(np.std(y_train_vals))
        if y_std < 1e-6:
            y_std = 1.0
        yields_norm = ((yields - y_mean) / y_std).astype(np.float32)
        _log(f"  Yield normalization: mean={y_mean:.1f}, std={y_std:.1f}")
    else:
        yields_norm = yields.astype(np.float32)
        y_mean, y_std = 0.0, 1.0

    # ── To tensors ───────────────────────────────────────────────────────
    train_w, train_wm = _to(weather_x[idx_train], device), _to(weather_mask[idx_train], device)
    train_s, train_sm = _to(sat_x[idx_train], device), _to(sat_mask[idx_train], device)
    train_y = _to(yields_norm[idx_train], device)

    val_w, val_wm = _to(weather_x[idx_val], device), _to(weather_mask[idx_val], device)
    val_s, val_sm = _to(sat_x[idx_val], device), _to(sat_mask[idx_val], device)
    val_y = _to(yields_norm[idx_val], device)

    adj_t = torch.tensor(adj_np, dtype=torch.float32, device=device)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Train Backbone (Informer + GAT with MLP head)
    # ══════════════════════════════════════════════════════════════════════
    _log(f"\n{'='*60}")
    _log("STAGE 1: Training backbone (Informer + GAT with MLP head)...")
    _log(f"{'='*60}")

    model_kwargs = {
        "weather_input_dim": n_weather_feats,
        "sat_input_dim": n_sat_feats,
        "weather_d_model": model_cfg["weather_d_model"],
        "sat_d_model": model_cfg["sat_d_model"],
        "weather_heads": model_cfg["weather_heads"],
        "sat_heads": model_cfg["sat_heads"],
        "weather_layers": model_cfg["weather_layers"],
        "sat_layers": model_cfg["sat_layers"],
        "weather_d_ff": model_cfg["weather_d_ff"],
        "sat_d_ff": model_cfg["sat_d_ff"],
        "dropout": model_cfg["dropout"],
        "gat_hidden": model_cfg["gat_hidden"],
        "gat_heads": model_cfg["gat_heads"],
        "gat_layers": model_cfg["gat_layers"],
        "weather_distil": model_cfg["weather_distil"],
        "sat_distil": model_cfg["sat_distil"],
    }
    model = DualChannelInformerGAT(**model_kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"  Backbone params: total={total_params:,}, trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Loss function ────────────────────────────────────────────────────
    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0)
        _log(f"  Loss: HuberLoss (delta=1.0)")
    else:
        criterion = nn.MSELoss()
        _log(f"  Loss: MSELoss")

    # ── LR Scheduler ─────────────────────────────────────────────────────
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        _log(f"  Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min={lr*0.01:.1e})")
    else:
        scheduler = None

    # ── Backbone Training loop ───────────────────────────────────────────
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    t0 = time.perf_counter()

    _log(f"\nTraining backbone for {epochs} epochs...\n")
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(train_w, train_wm, train_s, train_sm, adj_t)
        loss = criterion(pred, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_w, val_wm, val_s, val_sm, adj_t)
            val_loss = criterion(val_pred, val_y).item()
            if not np.isfinite(val_loss):
                _log(f"⚠ Val loss non-finite at epoch {ep}, stopping to prevent NaN.")
                break

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        # Track best model (but do NOT stop early).
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            elapsed = time.perf_counter() - t0
            _log(
                f"  epoch {ep:04d}/{epochs}  "
                f"train_loss={loss.item():.6f}  val_loss={val_loss:.6f}  "
                f"best_ep={best_epoch}  [{elapsed:.1f}s]"
            )

    backbone_time = time.perf_counter() - t0
    _log(f"\nBackbone training complete in {backbone_time:.1f}s "
         f"({len(train_losses)} epochs, best epoch={best_epoch})")

    # ── Restore best backbone ────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Extract GAT Embeddings → Train XGBoost
    # ══════════════════════════════════════════════════════════════════════
    _log(f"\n{'='*60}")
    _log("STAGE 2: Extracting GAT embeddings → Training XGBoost head...")
    _log(f"{'='*60}")

    # ── Extract embeddings for all seasons ────────────────────────────────
    with torch.no_grad():
        all_w = _to(weather_x, device)
        all_wm = _to(weather_mask, device)
        all_s = _to(sat_x, device)
        all_sm = _to(sat_mask, device)
        _, all_node_feat = model(all_w, all_wm, all_s, all_sm, adj_t)
    # all_node_feat: [S, N, D] — post-GAT embeddings
    embeddings = all_node_feat.detach().cpu().numpy()  # [S, N, D]
    emb_dim = embeddings.shape[-1]
    _log(f"  Extracted embeddings: shape={embeddings.shape}, dim={emb_dim}")

    # ── Flatten to [samples, features] for XGBoost ───────────────────────
    # Each (season, district) pair = one sample.
    X_train = embeddings[idx_train].reshape(-1, emb_dim)   # [S_train * N, D]
    y_train_xgb = yields[idx_train].ravel()                # [S_train * N]

    X_val = embeddings[idx_val].reshape(-1, emb_dim)       # [S_val * N, D]
    y_val_xgb = yields[idx_val].ravel()                    # [S_val * N]

    X_all = embeddings.reshape(-1, emb_dim)                # [S * N, D]

    _log(f"  XGBoost train samples: {X_train.shape[0]}")
    _log(f"  XGBoost val samples:   {X_val.shape[0]}")
    _log(f"  XGBoost total samples: {X_all.shape[0]}")

    # ── Train XGBoost ────────────────────────────────────────────────────
    xgb_params = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
    }
    _log(f"  XGBoost params: {xgb_params}")

    xgb_model = XGBRegressor(**xgb_params)

    t1 = time.perf_counter()
    xgb_model.fit(
        X_train, y_train_xgb,
        eval_set=[(X_val, y_val_xgb)],
        verbose=False,
    )
    xgb_time = time.perf_counter() - t1
    _log(f"  XGBoost training complete in {xgb_time:.1f}s")

    best_iteration = xgb_model.best_iteration if hasattr(xgb_model, "best_iteration") else args.xgb_n_estimators
    _log(f"  XGBoost best iteration: {best_iteration}")

    # ── Predict with XGBoost on all data ─────────────────────────────────
    all_pred_flat = xgb_model.predict(X_all)  # [S * N]
    all_pred_np = all_pred_flat.reshape(n_seasons, n_districts)  # [S, N]

    train_pred = all_pred_np[idx_train]
    val_pred = all_pred_np[idx_val]

    train_y_np = yields[idx_train]
    val_y_np = yields[idx_val]

    train_m = _metrics(train_y_np.ravel(), train_pred.ravel())
    val_m = _metrics(val_y_np.ravel(), val_pred.ravel())

    has_test = len(idx_test) > 0
    if has_test:
        test_pred = all_pred_np[idx_test]
        test_y_np = yields[idx_test]
        test_m = _metrics(test_y_np.ravel(), test_pred.ravel())
    else:
        test_pred = np.empty((0, n_districts), dtype=np.float32)
        test_y_np = np.empty((0, n_districts), dtype=np.float32)
        test_m = {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0}

    _log(f"\n{'='*60}")
    _log(f"RESULTS (Backbone best_ep={best_epoch}, XGBoost head):")
    _log(f"  Train — RMSE={train_m['rmse']:.2f}, MAE={train_m['mae']:.2f}, "
         f"MAPE={train_m['mape']:.2f}%, R²={train_m['r2']:.4f}")
    _log(f"  Val   — RMSE={val_m['rmse']:.2f}, MAE={val_m['mae']:.2f}, "
         f"MAPE={val_m['mape']:.2f}%, R²={val_m['r2']:.4f}")
    if has_test:
        _log(f"  Test  — RMSE={test_m['rmse']:.2f}, MAE={test_m['mae']:.2f}, "
             f"MAPE={test_m['mape']:.2f}%, R²={test_m['r2']:.4f}")
    else:
        _log(f"  Test  — (no test set)")
    _log(f"{'='*60}")

    # ── Build split labels ───────────────────────────────────────────────
    split_labels = np.array([""] * n_seasons, dtype=object)
    for i in idx_train:
        split_labels[i] = "train"
    for i in idx_val:
        split_labels[i] = "val"
    for i in idx_test:
        split_labels[i] = "test"

    # ── Save all analysis artifacts to run_X folder ──────────────────────
    _log(f"\nSaving analysis to {run_dir}/...")

    # 1. Metrics
    _save_metrics(train_m, val_m, test_m, run_dir)

    # 2. Predictions
    _save_predictions(
        district_ids, district_names, state_names,
        season_years, split_labels,
        yields, all_pred_np, run_dir,
    )

    # 3. Training curves (backbone)
    _save_training_curves(train_losses, val_losses, run_dir)

    # 4. Error analysis
    _save_error_analysis(
        np.concatenate([train_y_np.ravel(), val_y_np.ravel(), test_y_np.ravel()]),
        np.concatenate([train_pred.ravel(), val_pred.ravel(), test_pred.ravel()]),
        run_dir,
    )

    # 5. Model parameters (backbone + XGBoost)
    _save_model_params(model, model_kwargs, cfg, xgb_params, run_dir)

    # 6. Accuracy classification (MAPE-based)
    _save_accuracy_classification(
        district_ids, district_names, state_names,
        season_years, split_labels,
        yields, all_pred_np, run_dir,
    )

    # 7. District-wise trend analysis with precision/recall
    _save_trend_analysis(
        district_ids, district_names, state_names,
        season_years, split_labels,
        yields, all_pred_np, run_dir,
    )

    _log(f"\n✓ All analysis artifacts saved to {run_dir}/")
    _log(f"  Total time: backbone={backbone_time:.1f}s + xgboost={xgb_time:.1f}s = {backbone_time+xgb_time:.1f}s")
    _log("Done.")


if __name__ == "__main__":
    main()
