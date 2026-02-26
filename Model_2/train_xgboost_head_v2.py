#!/usr/bin/env python3
"""
train_xgboost_head_v2.py — Informer + GAT Backbone → XGBoost Head (v2)
=======================================================================
Improved version with 6 anti-overfitting changes vs v1:
  1. Backbone early stopping (patience-based)
  2. Higher dropout (0.3 default, override via --dropout)
  3. Heavily regularized XGBoost (shallow trees, L1/L2, early stopping)
  4. PCA dimensionality reduction on embeddings
  5. Feature augmentation (district yield history + raw embedding stats)
  6. Noise injection on train embeddings for data augmentation

Usage:
    python Model_2/train_xgboost_head_v2.py
    python Model_2/train_xgboost_head_v2.py --epochs 100 --pca-components 10
    python Model_2/train_xgboost_head_v2.py --no-pca --no-noise

Analysis artifacts are auto-saved to Model_2/analysis/run_X/.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from sklearn.decomposition import PCA

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError(
        "xgboost is required for this script. Install it via:\n"
        "  pip install xgboost"
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))
from informer_gat_model import DualChannelInformerGAT  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (identical to v1)
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(f"[XGB-v2] {msg}", flush=True)


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


def _season_split(
    years: Sequence[int], mode: str, seed: int,
    custom_train: Optional[List[int]] = None,
    custom_val: Optional[List[int]] = None,
    custom_test: Optional[List[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
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


def _fit_scaler(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
# Analysis generators (reused from v1 with minor adjustments)
# ═══════════════════════════════════════════════════════════════════════════════

def _save_metrics(train_m, val_m, test_m, out_dir):
    data = {"train": train_m, "val": val_m, "test": test_m}
    path = out_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(f"  → {path}")


def _save_predictions(
    district_ids, district_names, state_names,
    season_years, split_labels, y_true, y_pred, out_dir,
):
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


def _save_training_curves(train_losses, val_losses, best_epoch, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="Train Loss (Backbone)", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss (Backbone)", color="#F44336", linewidth=1.5)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#4CAF50", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Backbone Training Curves (v2 — Early Stopping)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  → {path}")


def _save_error_analysis(y_true_all, y_pred_all, out_dir):
    errors = y_pred_all - y_true_all
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(y_true_all, y_pred_all, alpha=0.4, s=15, c="#1976D2", edgecolors="none")
    lo = min(y_true_all.min(), y_pred_all.min()) * 0.9
    hi = max(y_true_all.max(), y_pred_all.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual Yield (kg/ha)", fontsize=11)
    ax.set_ylabel("Predicted Yield (kg/ha)", fontsize=11)
    ax.set_title("Actual vs Predicted (XGBoost v2)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.hist(errors, bins=40, color="#66BB6A", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Error (kg/ha)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Distribution (XGBoost v2)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "error_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  → {path}")


def _save_model_params(model, model_kwargs, cfg, xgb_params, v2_config, out_dir):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_info = []
    for name, param in model.named_parameters():
        layer_info.append({
            "name": name,
            "shape": list(param.shape),
            "params": int(param.numel()),
            "trainable": bool(param.requires_grad),
        })
    data = {
        "architecture": "DualChannelInformerGAT (backbone) + XGBoost (head) [v2]",
        "backbone_total_params": int(total),
        "backbone_trainable_params": int(trainable),
        "backbone_hyperparameters": model_kwargs,
        "xgboost_hyperparameters": xgb_params,
        "v2_improvements": v2_config,
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


# ── Accuracy Classification ────────────────────────────────────────────────

def _classify_accuracy(mape_pct):
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


def _save_accuracy_chart(df, title_suffix, out_dir, file_prefix):
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
    district_ids, district_names, state_names,
    season_years, split_labels, y_true, y_pred, out_dir,
):
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

    csv_path = out_dir / "accuracy_classification_all.csv"
    df.to_csv(csv_path, index=False)
    _log(f"  → {csv_path}")
    _save_accuracy_chart(df, " (All)", out_dir, "accuracy_classification_all")

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


# ── Trend Analysis ─────────────────────────────────────────────────────────

def _compute_trend_report(df):
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
        report[d] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1_score": round(f1, 4), "support": tp + fn,
            "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
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
        "macro_avg": {"precision": round(float(macro_prec), 4),
                      "recall": round(float(macro_rec), 4),
                      "f1_score": round(float(macro_f1), 4)},
        "weighted_avg": {"precision": round(float(w_prec), 4),
                         "recall": round(float(w_rec), 4),
                         "f1_score": round(float(w_f1), 4)},
    }


def _save_trend_visualization(df, report, title_suffix, out_dir, file_prefix):
    directions = ["increase", "decrease", "stable"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    conf = np.zeros((3, 3), dtype=int)
    for i, ad in enumerate(directions):
        for j, pd_ in enumerate(directions):
            conf[i, j] = int(((df["actual_direction"] == ad) & (df["predicted_direction"] == pd_)).sum())
    im = ax.imshow(conf, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
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
    ax.set_ylim(0, 1.15); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    png_path = out_dir / f"{file_prefix}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  → {png_path}")


def _save_trend_analysis(
    district_ids, district_names, state_names,
    season_years, split_labels, y_true, y_pred, out_dir,
):
    import pandas as pd
    sorted_idx = np.argsort(season_years)
    sorted_years = season_years[sorted_idx]
    sorted_true = y_true[sorted_idx]
    sorted_pred = y_pred[sorted_idx]
    sorted_splits = split_labels[sorted_idx]
    year_to_split = {int(sorted_years[i]): sorted_splits[i] for i in range(len(sorted_years))}
    THRESHOLD_PCT = 0.5
    rows = []
    for ti in range(len(sorted_years) - 1):
        year_from = int(sorted_years[ti])
        year_to = int(sorted_years[ti + 1])
        transition_split = year_to_split.get(year_to, "unknown")
        for di in range(len(district_ids)):
            actual_from, actual_to = sorted_true[ti, di], sorted_true[ti + 1, di]
            pred_from, pred_to = sorted_pred[ti, di], sorted_pred[ti + 1, di]
            actual_change_pct = ((actual_to - actual_from) / max(abs(actual_from), 1e-6)) * 100
            actual_dir = "increase" if actual_change_pct > THRESHOLD_PCT else ("decrease" if actual_change_pct < -THRESHOLD_PCT else "stable")
            pred_change_pct = ((pred_to - pred_from) / max(abs(pred_from), 1e-6)) * 100
            pred_dir = "increase" if pred_change_pct > THRESHOLD_PCT else ("decrease" if pred_change_pct < -THRESHOLD_PCT else "stable")
            rows.append({
                "split": transition_split,
                "district_id": str(district_ids[di]),
                "district_name": str(district_names[di]),
                "state_name": str(state_names[di]),
                "year_from": year_from, "year_to": year_to,
                "actual_yield_from": float(actual_from), "actual_yield_to": float(actual_to),
                "actual_change_pct": float(actual_change_pct), "actual_direction": actual_dir,
                "predicted_yield_from": float(pred_from), "predicted_yield_to": float(pred_to),
                "predicted_change_pct": float(pred_change_pct), "predicted_direction": pred_dir,
                "direction_match": bool(actual_dir == pred_dir),
            })
    df = pd.DataFrame(rows)
    csv_path = out_dir / "trend_analysis_all.csv"
    df.to_csv(csv_path, index=False)
    _log(f"  → {csv_path}")
    all_report = _compute_trend_report(df)
    report_path = out_dir / "trend_analysis_all.json"
    with open(report_path, "w") as f:
        json.dump(all_report, f, indent=2)
    _log(f"  → {report_path}")
    _save_trend_visualization(df, all_report, " (All)", out_dir, "trend_analysis_all")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        if len(split_df) == 0:
            _log(f"  ⚠ No trend transitions for split '{split_name}', skipping.")
            continue
        split_csv = out_dir / f"trend_analysis_{split_name}.csv"
        split_df.to_csv(split_csv, index=False)
        _log(f"  → {split_csv}")
        split_report = _compute_trend_report(split_df)
        split_json = out_dir / f"trend_analysis_{split_name}.json"
        with open(split_json, "w") as f:
            json.dump(split_report, f, indent=2)
        _log(f"  → {split_json}")
        _save_trend_visualization(
            split_df, split_report, f" ({split_name.title()})",
            out_dir, f"trend_analysis_{split_name}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NEW v2: Feature engineering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_district_history_features(
    yields: np.ndarray, idx_train: np.ndarray, n_districts: int,
) -> np.ndarray:
    """
    Build per-district historical yield statistics from training years only.
    Returns [N, 3] array: (mean, std, trend_slope) per district.
    """
    train_yields = yields[idx_train]  # [S_train, N]
    d_mean = train_yields.mean(axis=0)  # [N]
    d_std = train_yields.std(axis=0)    # [N]
    d_std = np.where(d_std < 1e-6, 1.0, d_std)

    # Linear trend slope per district over training years.
    n_train = len(idx_train)
    x = np.arange(n_train, dtype=np.float64)
    x_mean = x.mean()
    d_slope = np.zeros(n_districts, dtype=np.float64)
    for di in range(n_districts):
        y = train_yields[:, di].astype(np.float64)
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom > 1e-12:
            d_slope[di] = np.sum((x - x_mean) * (y - y_mean)) / denom

    return np.stack([d_mean, d_std, d_slope], axis=1).astype(np.float32)  # [N, 3]


def _augment_with_noise(
    X: np.ndarray, n_copies: int, noise_scale: float, rng: np.random.RandomState,
) -> np.ndarray:
    """Add Gaussian noise copies to training data for augmentation."""
    augmented = [X]
    for _ in range(n_copies):
        std_per_feat = X.std(axis=0)
        std_per_feat = np.where(std_per_feat < 1e-8, 1e-8, std_per_feat)
        noise = rng.normal(0, noise_scale, size=X.shape) * std_per_feat
        augmented.append(X + noise)
    return np.vstack(augmented)


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Two-Stage Pipeline (v2)
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Informer+GAT → XGBoost head (v2, anti-overfitting)."
    )
    parser.add_argument("--config", type=str, default="Model_2/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    # v2-specific args.
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Backbone dropout (v2 default: 0.3, was 0.2 in v1).")
    parser.add_argument("--pca-components", type=int, default=10,
                        help="Number of PCA components for embedding reduction.")
    parser.add_argument("--no-pca", action="store_true",
                        help="Disable PCA dimensionality reduction.")
    parser.add_argument("--noise-copies", type=int, default=3,
                        help="Number of augmented noise copies of training data.")
    parser.add_argument("--noise-scale", type=float, default=0.1,
                        help="Gaussian noise scale (fraction of feature std).")
    parser.add_argument("--no-noise", action="store_true",
                        help="Disable noise injection augmentation.")
    # XGBoost args (v2 regularized defaults).
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)
    parser.add_argument("--xgb-min-child-weight", type=int, default=10)
    parser.add_argument("--xgb-reg-alpha", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=5.0)
    parser.add_argument("--xgb-patience", type=int, default=30)
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
    patience = train_cfg.get("patience", 50)

    dataset_path = Path(args.dataset or data_cfg["dataset_file"])
    analysis_root = Path(cfg["output"]["analysis_dir"])
    run_dir = _next_run_dir(analysis_root)
    _log(f"Run output directory: {run_dir}")

    _set_seed(seed)
    device = _pick_device(device_name)
    _log(f"Device: {device}, Seed: {seed}")

    # ── Load dataset ─────────────────────────────────────────────────────
    _log(f"Loading dataset from {dataset_path}...")
    ds = np.load(dataset_path, allow_pickle=True)
    weather_x = ds["weather_x"]
    weather_mask = ds["weather_mask"]
    sat_x = ds["sat_x"]
    sat_mask = ds["sat_mask"]
    yields = ds["yields"]
    adj_np = ds["adjacency"]
    district_ids = ds["district_ids"]
    season_years = ds["season_years"]
    district_names = ds["district_names"]
    state_names = ds["state_names"]

    n_weather_feats = weather_x.shape[-1]
    n_sat_feats = sat_x.shape[-1]
    n_districts = weather_x.shape[1]
    n_seasons = weather_x.shape[0]

    _log(f"  Seasons: {list(season_years)} ({n_seasons}), Districts: {n_districts}")

    # ── Exclude states ───────────────────────────────────────────────────
    exclude_states = split_cfg.get("exclude_states", [])
    if exclude_states:
        keep_mask = np.array([s not in exclude_states for s in state_names])
        keep_idx = np.where(keep_mask)[0]
        _log(f"  Excluding {exclude_states}: {n_districts - len(keep_idx)} removed, {len(keep_idx)} kept")
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

    # ── Scale features ───────────────────────────────────────────────────
    w_mean, w_std = _fit_scaler(weather_x[idx_train], weather_mask[idx_train])
    s_mean, s_std = _fit_scaler(sat_x[idx_train], sat_mask[idx_train])
    weather_x = _apply_scaler(weather_x, w_mean, w_std)
    sat_x = _apply_scaler(sat_x, s_mean, s_std)

    # ── Yield normalization ──────────────────────────────────────────────
    if normalize_yield:
        y_train_vals = yields[idx_train].ravel()
        y_mean = float(np.mean(y_train_vals))
        y_std = float(np.std(y_train_vals))
        if y_std < 1e-6:
            y_std = 1.0
        yields_norm = ((yields - y_mean) / y_std).astype(np.float32)
        _log(f"  Yield norm: mean={y_mean:.1f}, std={y_std:.1f}")
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
    # STAGE 1: Train Backbone with EARLY STOPPING + HIGHER DROPOUT
    # ══════════════════════════════════════════════════════════════════════
    _log(f"\n{'='*60}")
    _log("STAGE 1: Training backbone (v2: early stopping + dropout={:.1f})...".format(args.dropout))
    _log(f"{'='*60}")

    # IMPROVEMENT 2: Override dropout.
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
        "dropout": args.dropout,  # v2: higher dropout
        "gat_hidden": model_cfg["gat_hidden"],
        "gat_heads": model_cfg["gat_heads"],
        "gat_layers": model_cfg["gat_layers"],
        "weather_distil": model_cfg["weather_distil"],
        "sat_distil": model_cfg["sat_distil"],
    }
    model = DualChannelInformerGAT(**model_kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    _log(f"  Backbone params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0)
        _log(f"  Loss: HuberLoss (delta=1.0)")
    else:
        criterion = nn.MSELoss()
        _log(f"  Loss: MSELoss")

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
    else:
        scheduler = None

    # IMPROVEMENT 1: Early stopping.
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    t0 = time.perf_counter()

    _log(f"  Training for up to {epochs} epochs (patience={patience})...\n")
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
                _log(f"  ⚠ Val loss non-finite at epoch {ep}, stopping.")
                break

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if ep == 1 or ep % 10 == 0 or ep == epochs or epochs_no_improve >= patience:
            elapsed = time.perf_counter() - t0
            _log(f"  epoch {ep:04d}/{epochs}  train={loss.item():.6f}  "
                 f"val={val_loss:.6f}  best_ep={best_epoch}  [{elapsed:.1f}s]")

        # Note: early stopping disabled — train for all epochs, restore best.

    backbone_time = time.perf_counter() - t0
    actual_epochs = len(train_losses)
    _log(f"\nBackbone done in {backbone_time:.1f}s ({actual_epochs} epochs, best={best_epoch})")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Extract Embeddings → PCA → Augment → XGBoost
    # ══════════════════════════════════════════════════════════════════════
    _log(f"\n{'='*60}")
    _log("STAGE 2: Embeddings → PCA → Feature Augmentation → XGBoost...")
    _log(f"{'='*60}")

    # ── Extract embeddings ───────────────────────────────────────────────
    with torch.no_grad():
        all_w = _to(weather_x, device)
        all_wm = _to(weather_mask, device)
        all_s = _to(sat_x, device)
        all_sm = _to(sat_mask, device)
        _, all_node_feat = model(all_w, all_wm, all_s, all_sm, adj_t)
    embeddings = all_node_feat.detach().cpu().numpy()  # [S, N, D]
    emb_dim = embeddings.shape[-1]
    _log(f"  Raw embeddings: shape={embeddings.shape}, dim={emb_dim}")

    # Flatten to [S*N, D].
    emb_flat = embeddings.reshape(-1, emb_dim)  # [S*N, D]

    # ── IMPROVEMENT 4: PCA dimensionality reduction ──────────────────────
    use_pca = not args.no_pca
    if use_pca:
        n_components = min(args.pca_components, emb_dim, len(idx_train) * n_districts)
        _log(f"  PCA: reducing {emb_dim} → {n_components} components (fit on train only)")
        train_emb_flat = embeddings[idx_train].reshape(-1, emb_dim)
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(train_emb_flat)
        explained = pca.explained_variance_ratio_.sum() * 100
        _log(f"  PCA explained variance: {explained:.1f}%")
        emb_flat = pca.transform(emb_flat)
        emb_dim_after = emb_flat.shape[1]
    else:
        emb_dim_after = emb_dim
        _log(f"  PCA: disabled")

    # ── IMPROVEMENT 5: Feature augmentation (district history) ───────────
    district_hist = _build_district_history_features(yields, idx_train, n_districts)
    _log(f"  District history features: shape={district_hist.shape} (mean, std, slope)")

    # Tile district_hist across seasons: [S*N, 3].
    hist_tiled = np.tile(district_hist, (n_seasons, 1))  # [S*N, 3]
    X_all = np.hstack([emb_flat, hist_tiled])  # [S*N, D'+3]
    feat_dim = X_all.shape[1]
    _log(f"  Final feature dim: {feat_dim} (embeddings={emb_dim_after} + history=3)")

    # Split.
    n_per_season = n_districts
    X_train = X_all[np.concatenate([np.arange(i * n_per_season, (i + 1) * n_per_season) for i in idx_train])]
    y_train_xgb = yields[idx_train].ravel()
    X_val = X_all[np.concatenate([np.arange(i * n_per_season, (i + 1) * n_per_season) for i in idx_val])]
    y_val_xgb = yields[idx_val].ravel()

    _log(f"  XGBoost train: {X_train.shape[0]} samples, val: {X_val.shape[0]} samples")

    # ── IMPROVEMENT 6: Noise injection augmentation ──────────────────────
    use_noise = not args.no_noise
    if use_noise and args.noise_copies > 0:
        rng = np.random.RandomState(seed)
        n_orig = X_train.shape[0]
        X_train_aug = _augment_with_noise(X_train, args.noise_copies, args.noise_scale, rng)
        y_train_aug = np.tile(y_train_xgb, args.noise_copies + 1)
        _log(f"  Noise augmentation: {n_orig} → {X_train_aug.shape[0]} samples "
             f"({args.noise_copies} copies, σ={args.noise_scale})")
    else:
        X_train_aug = X_train
        y_train_aug = y_train_xgb
        _log(f"  Noise augmentation: disabled")

    # ── IMPROVEMENT 3: Regularized XGBoost with early stopping ───────────
    xgb_params = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "min_child_weight": args.xgb_min_child_weight,
        "reg_alpha": args.xgb_reg_alpha,
        "reg_lambda": args.xgb_reg_lambda,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": args.xgb_patience,
    }
    _log(f"  XGBoost params: {xgb_params}")

    xgb_model = XGBRegressor(**xgb_params)
    t1 = time.perf_counter()
    xgb_model.fit(
        X_train_aug, y_train_aug,
        eval_set=[(X_val, y_val_xgb)],
        verbose=False,
    )
    xgb_time = time.perf_counter() - t1

    best_xgb_iter = getattr(xgb_model, "best_iteration", args.xgb_n_estimators)
    _log(f"  XGBoost training: {xgb_time:.1f}s, best iteration: {best_xgb_iter}")

    # ── Predict ──────────────────────────────────────────────────────────
    all_pred_flat = xgb_model.predict(X_all)
    all_pred_np = all_pred_flat.reshape(n_seasons, n_districts)

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
    _log(f"RESULTS (v2: backbone_ep={best_epoch}/{actual_epochs}, XGBoost iter={best_xgb_iter}):")
    _log(f"  Train — RMSE={train_m['rmse']:.2f}, MAE={train_m['mae']:.2f}, "
         f"MAPE={train_m['mape']:.2f}%, R²={train_m['r2']:.4f}")
    _log(f"  Val   — RMSE={val_m['rmse']:.2f}, MAE={val_m['mae']:.2f}, "
         f"MAPE={val_m['mape']:.2f}%, R²={val_m['r2']:.4f}")
    if has_test:
        _log(f"  Test  — RMSE={test_m['rmse']:.2f}, MAE={test_m['mae']:.2f}, "
             f"MAPE={test_m['mape']:.2f}%, R²={test_m['r2']:.4f}")
    _log(f"{'='*60}")

    # ── Build split labels ───────────────────────────────────────────────
    split_labels = np.array([""] * n_seasons, dtype=object)
    for i in idx_train:
        split_labels[i] = "train"
    for i in idx_val:
        split_labels[i] = "val"
    for i in idx_test:
        split_labels[i] = "test"

    # ── v2 config for model_params ───────────────────────────────────────
    v2_config = {
        "dropout_override": args.dropout,
        "backbone_early_stopping_patience": patience,
        "backbone_actual_epochs": actual_epochs,
        "backbone_best_epoch": best_epoch,
        "pca_enabled": use_pca,
        "pca_components": args.pca_components if use_pca else None,
        "pca_explained_variance_pct": float(explained) if use_pca else None,
        "noise_injection_enabled": use_noise,
        "noise_copies": args.noise_copies if use_noise else 0,
        "noise_scale": args.noise_scale if use_noise else 0,
        "train_samples_original": int(len(idx_train) * n_districts),
        "train_samples_augmented": int(X_train_aug.shape[0]),
        "feature_dim_final": feat_dim,
    }

    # ── Save all analysis artifacts ──────────────────────────────────────
    _log(f"\nSaving analysis to {run_dir}/...")
    _save_metrics(train_m, val_m, test_m, run_dir)
    _save_predictions(
        district_ids, district_names, state_names,
        season_years, split_labels, yields, all_pred_np, run_dir,
    )
    _save_training_curves(train_losses, val_losses, best_epoch, run_dir)
    _save_error_analysis(
        np.concatenate([train_y_np.ravel(), val_y_np.ravel(), test_y_np.ravel()]),
        np.concatenate([train_pred.ravel(), val_pred.ravel(), test_pred.ravel()]),
        run_dir,
    )
    _save_model_params(model, model_kwargs, cfg, xgb_params, v2_config, run_dir)
    _save_accuracy_classification(
        district_ids, district_names, state_names,
        season_years, split_labels, yields, all_pred_np, run_dir,
    )
    _save_trend_analysis(
        district_ids, district_names, state_names,
        season_years, split_labels, yields, all_pred_np, run_dir,
    )

    _log(f"\n✓ All analysis artifacts saved to {run_dir}/")
    _log(f"  Total time: backbone={backbone_time:.1f}s + xgboost={xgb_time:.1f}s = {backbone_time+xgb_time:.1f}s")
    _log("Done.")


if __name__ == "__main__":
    main()
