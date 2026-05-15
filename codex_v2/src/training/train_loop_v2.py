from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from codex_v2.src.data.build_dataset_v2 import DatasetBundle, apply_target_transform
from codex_v2.src.eval.metrics_v2 import regression_metrics
from codex_v2.src.models.dual_channel_transformer_gat_v2 import DualChannelTransformerGATV2
from codex_v2.src.training.losses_v2 import build_loss


@dataclass
class TrainingOutput:
    pred_target: np.ndarray
    pred_raw: np.ndarray
    best_epoch: int
    epochs_ran: int
    model_total_params: int
    model_trainable_params: int
    history: List[Dict[str, float]]
    train_seconds: float


class IndexDataset(Dataset):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(device_str: str) -> torch.device:
    if str(device_str).strip().lower() != "auto":
        try:
            return torch.device(device_str)
        except Exception:
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"total": total, "trainable": trainable}


def _build_model(model_cfg: Dict[str, object], bundle: DatasetBundle) -> DualChannelTransformerGATV2:
    op_vocab_size = int(max(bundle.sample_opdate_idx.tolist()) + 1) if len(bundle.sample_opdate_idx) > 0 else 1
    return DualChannelTransformerGATV2(
        weather_input_dim=int(bundle.weather_x.shape[-1]),
        sat_input_dim=int(bundle.sat_x.shape[-1]),
        weather_d_model=int(model_cfg["weather_d_model"]),
        sat_d_model=int(model_cfg["sat_d_model"]),
        weather_heads=int(model_cfg["weather_heads"]),
        sat_heads=int(model_cfg["sat_heads"]),
        weather_layers=int(model_cfg["weather_layers"]),
        sat_layers=int(model_cfg["sat_layers"]),
        weather_d_ff=int(model_cfg["weather_d_ff"]),
        sat_d_ff=int(model_cfg["sat_d_ff"]),
        dropout=float(model_cfg["dropout"]),
        fusion_hidden=int(model_cfg["fusion_hidden"]),
        gat_hidden=int(model_cfg["gat_hidden"]),
        gat_heads=int(model_cfg["gat_heads"]),
        gat_layers=int(model_cfg["gat_layers"]),
        max_seq_len_weather=int(model_cfg.get("max_seq_len_weather", 64)),
        max_seq_len_sat=int(model_cfg.get("max_seq_len_sat", 64)),
        opdate_embedding_dim=int(model_cfg.get("opdate_embedding_dim", 16)),
        opdate_vocab_size=op_vocab_size,
        fusion_mode=str(model_cfg.get("fusion_mode", "concat_gate")),
        pooling_mode_weather=str(model_cfg.get("pooling_mode_weather", "mean")),
        pooling_mode_sat=str(model_cfg.get("pooling_mode_sat", "mean")),
        cross_attention_level=str(model_cfg.get("cross_attention_level", "pooled")),
        cross_attention_heads=int(model_cfg.get("cross_attention_heads", 4)),
        head_mode=str(model_cfg.get("head_mode", "scalar")),
    )


def _warmup_cosine_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _neutral_band_value(
    train_delta: np.ndarray,
    training_cfg: Dict[str, object],
) -> float:
    abs_delta = np.abs(train_delta.astype(np.float32)).ravel()
    abs_delta = abs_delta[np.isfinite(abs_delta)]
    if abs_delta.size == 0:
        return 50.0
    q = float(training_cfg.get("neutral_quantile", 0.10))
    floor = float(training_cfg.get("neutral_floor", 25.0))
    cap = float(training_cfg.get("neutral_cap", 125.0))
    eps = float(np.quantile(abs_delta, min(max(q, 0.0), 0.49)))
    return float(min(max(eps, floor), cap))


def _delta_to_class(delta_raw: torch.Tensor, neutral_eps: float) -> torch.Tensor:
    cls = torch.full_like(delta_raw, 1, dtype=torch.long)
    cls = torch.where(delta_raw < -neutral_eps, torch.zeros_like(cls), cls)
    cls = torch.where(delta_raw > neutral_eps, torch.full_like(cls, 2), cls)
    return cls


def _sign_magnitude_loss(
    output,
    y_raw: torch.Tensor,
    target_mean: torch.Tensor,
    neutral_eps: float,
    class_loss_weight: float,
    magnitude_loss_weight: float,
    neutral_weight: float,
    mag_huber_delta: float,
) -> torch.Tensor:
    if output.class_logits is None or output.drop_logmag is None or output.rise_logmag is None:
        raise RuntimeError("sign_magnitude head requested, but model output is missing classifier/magnitude tensors.")

    delta_true = y_raw - target_mean.unsqueeze(0)
    cls_target = _delta_to_class(delta_true, neutral_eps=neutral_eps)

    ce = torch.nn.functional.cross_entropy(
        output.class_logits.reshape(-1, output.class_logits.shape[-1]),
        cls_target.reshape(-1),
        reduction="none",
    ).view_as(cls_target)
    ce_weights = torch.ones_like(ce)
    ce_weights = torch.where(cls_target == 1, torch.full_like(ce_weights, float(neutral_weight)), ce_weights)
    cls_loss = torch.mean(ce * ce_weights)

    mag_target = torch.log1p(torch.abs(delta_true))
    mag_terms: List[torch.Tensor] = []
    drop_mask = cls_target == 0
    rise_mask = cls_target == 2
    if torch.any(drop_mask):
        mag_terms.append(
            torch.nn.functional.huber_loss(
                output.drop_logmag[drop_mask],
                mag_target[drop_mask],
                delta=float(mag_huber_delta),
                reduction="mean",
            )
        )
    if torch.any(rise_mask):
        mag_terms.append(
            torch.nn.functional.huber_loss(
                output.rise_logmag[rise_mask],
                mag_target[rise_mask],
                delta=float(mag_huber_delta),
                reduction="mean",
            )
        )
    if mag_terms:
        mag_loss = torch.stack(mag_terms).mean()
    else:
        mag_loss = cls_loss.new_zeros(())

    return (float(class_loss_weight) * cls_loss) + (float(magnitude_loss_weight) * mag_loss)


def _predict_indices_raw(
    model: torch.nn.Module,
    idx: np.ndarray,
    weather_t: torch.Tensor,
    weather_mask_t: torch.Tensor,
    sat_t: torch.Tensor,
    sat_mask_t: torch.Tensor,
    y_raw_t: torch.Tensor,
    opdate_idx_t: torch.Tensor,
    adj_t: torch.Tensor,
    target_mean_t: torch.Tensor,
    inverse_target_fn,
) -> tuple[np.ndarray, np.ndarray]:
    if len(idx) == 0:
        return np.zeros((0, y_raw_t.shape[1]), dtype=np.float32), np.zeros((0, y_raw_t.shape[1]), dtype=np.float32)

    i_t = torch.tensor(idx, dtype=torch.long, device=weather_t.device)
    with torch.no_grad():
        output = model(
            weather_t[i_t],
            weather_mask_t[i_t],
            sat_t[i_t],
            sat_mask_t[i_t],
            adj_t,
            opdate_idx=opdate_idx_t[i_t],
        )
    y_true = y_raw_t[i_t].detach().cpu().numpy().astype(np.float32)

    if str(getattr(model, "head_mode", "scalar")).strip().lower() == "sign_magnitude":
        pred_raw = (output.pred_raw_delta + target_mean_t.unsqueeze(0)).detach().cpu().numpy().astype(np.float32)
        return y_true, pred_raw

    pred_target = output.pred.detach().cpu().numpy().astype(np.float32)
    pred_raw = inverse_target_fn(pred_target).astype(np.float32)
    return y_true, pred_raw


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))
    tn = int(np.sum((~yt) & (~yp)))
    return tp, fp, fn, tn


def train_model_v2(
    bundle: DatasetBundle,
    model_cfg: Dict[str, object],
    train_cfg: Dict[str, object],
    seed: int,
    out_dir: Path,
    mode: str,
    target_mode: str,
    horizon_days: int,
) -> TrainingOutput:
    optimization = train_cfg.get("optimization", {})
    training = train_cfg.get("training", {})
    runtime = train_cfg.get("runtime", {})

    epochs = int(training.get("epochs", 120))
    patience = int(training.get("patience", 30))
    batch_size = int(training.get("batch_size", 16))
    grad_clip = float(optimization.get("grad_clip", 1.0))
    lr = float(optimization.get("lr", 3e-4))
    weight_decay = float(optimization.get("weight_decay", 1e-2))
    warmup_pct = float(optimization.get("warmup_pct", 0.10))
    loss_name = str(optimization.get("loss", "huber"))
    rise_under_w = float(optimization.get("rise_under_w", 0.8))
    drop_miss_w = float(optimization.get("drop_miss_w", 0.2))
    huber_delta = float(optimization.get("huber_delta", 1.0))
    use_weighted_sampler = bool(training.get("use_weighted_sampler", False))
    sample_pos_weight = float(training.get("sample_pos_weight", 2.0))
    checkpoint_objective = str(training.get("checkpoint_objective", "rmse")).strip().lower()
    min_drop_recall = float(training.get("min_drop_recall", 0.777))
    neutral_eps = float(training.get("neutral_eps", -1.0))
    neutral_weight = float(training.get("neutral_weight", 0.35))
    class_loss_weight = float(training.get("class_loss_weight", 1.0))
    magnitude_loss_weight = float(training.get("magnitude_loss_weight", 1.0))
    magnitude_huber_delta = float(training.get("magnitude_huber_delta", 0.5))
    deterministic = bool(runtime.get("deterministic", True))
    device = choose_device(str(runtime.get("device", "auto")))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt = out_dir / "best_model.pt"
    training_log_path = out_dir / "training_log.jsonl"

    set_global_seed(seed=seed, deterministic=deterministic)

    model = _build_model(model_cfg=model_cfg, bundle=bundle).to(device)
    params = count_parameters(model)

    weather_t = torch.tensor(bundle.weather_x, dtype=torch.float32, device=device)
    weather_mask_t = torch.tensor(bundle.weather_mask, dtype=torch.float32, device=device)
    sat_t = torch.tensor(bundle.sat_x, dtype=torch.float32, device=device)
    sat_mask_t = torch.tensor(bundle.sat_mask, dtype=torch.float32, device=device)
    y_raw_t = torch.tensor(bundle.y_raw, dtype=torch.float32, device=device)
    y_target_t = torch.tensor(bundle.y_target, dtype=torch.float32, device=device)
    target_mean_t = torch.tensor(bundle.target_mean, dtype=torch.float32, device=device)
    opdate_idx_t = torch.tensor(bundle.sample_opdate_idx, dtype=torch.long, device=device)
    adj_t = torch.tensor(bundle.adjacency, dtype=torch.float32, device=device)

    head_mode = str(getattr(model, "head_mode", "scalar")).strip().lower()
    criterion = None
    if head_mode == "scalar":
        criterion = build_loss(
            loss_name=loss_name,
            huber_delta=huber_delta,
            rise_under_w=rise_under_w,
            drop_miss_w=drop_miss_w,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_sampler = None
    if use_weighted_sampler and len(bundle.train_idx) > 0:
        train_target = bundle.y_target[bundle.train_idx]
        pos_frac = (train_target > 0.0).mean(axis=1).astype(np.float32)
        med = float(np.median(pos_frac)) if len(pos_frac) > 0 else 0.0
        sample_w = 1.0 + sample_pos_weight * np.maximum(pos_frac - med, 0.0)
        sample_w = np.clip(sample_w, 1e-6, None)
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.double),
            num_samples=int(len(sample_w)),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(
        IndexDataset(bundle.train_idx),
        batch_size=batch_size,
        shuffle=bool(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    total_steps = max(1, epochs * max(1, len(train_loader)))
    warmup_steps = int(total_steps * warmup_pct)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_lambda(step, total_steps=total_steps, warmup_steps=warmup_steps),
    )

    best_val_rmse = float("inf")
    best_epoch_rmse = 0
    best_state_rmse: Optional[Dict[str, torch.Tensor]] = None
    best_epoch_obj = 0
    best_state_obj: Optional[Dict[str, torch.Tensor]] = None
    best_obj_primary = float("inf")
    best_obj_rmse = float("inf")
    found_obj_candidate = False
    wait = 0
    step = 0
    history: List[Dict[str, float]] = []
    train_delta = (bundle.y_raw[bundle.train_idx] - bundle.target_mean[None, :]).astype(np.float32)
    q75_train_delta = float(np.quantile(train_delta, 0.75)) if train_delta.size > 0 else float("inf")
    if head_mode == "sign_magnitude":
        neutral_eps = float(neutral_eps) if neutral_eps > 0.0 else _neutral_band_value(train_delta, training_cfg=training)

    run_t0 = time.perf_counter()

    pbar = tqdm(range(1, epochs + 1), desc=f"train[{mode}|{target_mode}|h{horizon_days}]", leave=True)
    with training_log_path.open("w") as log_fh:
        for epoch in pbar:
            model.train()
            train_losses: List[float] = []

            for idx_batch in train_loader:
                b = idx_batch.to(device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    weather_t[b],
                    weather_mask_t[b],
                    sat_t[b],
                    sat_mask_t[b],
                    adj_t,
                    opdate_idx=opdate_idx_t[b],
                )
                if head_mode == "sign_magnitude":
                    loss = _sign_magnitude_loss(
                        output=output,
                        y_raw=y_raw_t[b],
                        target_mean=target_mean_t,
                        neutral_eps=float(neutral_eps),
                        class_loss_weight=float(class_loss_weight),
                        magnitude_loss_weight=float(magnitude_loss_weight),
                        neutral_weight=float(neutral_weight),
                        mag_huber_delta=float(magnitude_huber_delta),
                    )
                else:
                    if criterion is None:
                        raise RuntimeError("Scalar head expected a pointwise criterion, but none was built.")
                    loss = criterion(output.pred, y_target_t[b])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()

                train_losses.append(float(loss.item()))
                step += 1

            model.eval()
            val_true_raw, val_pred_raw = _predict_indices_raw(
                model=model,
                idx=bundle.val_idx,
                weather_t=weather_t,
                weather_mask_t=weather_mask_t,
                sat_t=sat_t,
                sat_mask_t=sat_mask_t,
                y_raw_t=y_raw_t,
                opdate_idx_t=opdate_idx_t,
                adj_t=adj_t,
                target_mean_t=target_mean_t,
                inverse_target_fn=bundle.inverse_target_array,
            )
            val_metrics = regression_metrics(val_true_raw.ravel(), val_pred_raw.ravel())
            val_delta_true = (val_true_raw - bundle.target_mean[None, :]).astype(np.float32)
            val_delta_pred = (val_pred_raw - bundle.target_mean[None, :]).astype(np.float32)
            true_drop = val_delta_true < 0.0
            pred_drop = val_delta_pred < 0.0
            true_rise = val_delta_true > 0.0
            pred_rise = val_delta_pred > 0.0
            tp_d, fp_d, fn_d, tn_d = _confusion_binary(true_drop, pred_drop)
            tp_r, fp_r, fn_r, tn_r = _confusion_binary(true_rise, pred_rise)
            val_drop_recall = _safe_rate(tp_d, tp_d + fn_d)
            val_rise_recall = _safe_rate(tp_r, tp_r + fn_r)
            rise_under = np.maximum(val_delta_true - val_delta_pred, 0.0).astype(np.float32)
            rise_mask = val_delta_true > q75_train_delta
            val_rise_under_mae = (
                float(np.mean(rise_under[rise_mask])) if np.any(rise_mask) else float("nan")
            )

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            row = {
                "epoch": int(epoch),
                "seed": int(seed),
                "train_loss": train_loss,
                "val_rmse": float(val_metrics["rmse"]),
                "val_mae": float(val_metrics["mae"]),
                "val_r2": float(val_metrics["r2"]),
                "val_drop_recall": float(val_drop_recall),
                "val_rise_recall": float(val_rise_recall),
                "val_rise_under_mae": float(val_rise_under_mae),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()

            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_rmse=f"{val_metrics['rmse']:.2f}",
                val_r2=f"{val_metrics['r2']:.3f}",
                drop_rec=f"{val_drop_recall:.3f}" if np.isfinite(val_drop_recall) else "nan",
                rise_rec=f"{val_rise_recall:.3f}" if np.isfinite(val_rise_recall) else "nan",
                rise_u_mae=f"{val_rise_under_mae:.1f}" if np.isfinite(val_rise_under_mae) else "nan",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if val_metrics["rmse"] < best_val_rmse - 1e-8:
                best_val_rmse = float(val_metrics["rmse"])
                best_epoch_rmse = int(epoch)
                best_state_rmse = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
            if checkpoint_objective == "drop_constrained_rise":
                qualifies = bool(np.isfinite(val_drop_recall) and (val_drop_recall >= min_drop_recall))
                primary = float(val_rise_under_mae) if np.isfinite(val_rise_under_mae) else float("inf")
                rmse_now = float(val_metrics["rmse"])
                better = False
                if qualifies:
                    if not found_obj_candidate:
                        better = True
                    elif primary < best_obj_primary - 1e-8:
                        better = True
                    elif abs(primary - best_obj_primary) <= 1e-8 and rmse_now < best_obj_rmse - 1e-8:
                        better = True
                if better:
                    found_obj_candidate = True
                    best_obj_primary = primary
                    best_obj_rmse = rmse_now
                    best_epoch_obj = int(epoch)
                    best_state_obj = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    epochs_ran = len(history)
    best_epoch = best_epoch_rmse
    best_state = best_state_rmse
    if checkpoint_objective == "drop_constrained_rise" and found_obj_candidate and best_state_obj is not None:
        best_state = best_state_obj
        best_epoch = best_epoch_obj
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, model_ckpt)
    model.load_state_dict(best_state)
    model.eval()

    all_idx = np.arange(bundle.y_target.shape[0], dtype=np.int64)
    _, pred_raw = _predict_indices_raw(
        model=model,
        idx=all_idx,
        weather_t=weather_t,
        weather_mask_t=weather_mask_t,
        sat_t=sat_t,
        sat_mask_t=sat_mask_t,
        y_raw_t=y_raw_t,
        opdate_idx_t=opdate_idx_t,
        adj_t=adj_t,
        target_mean_t=target_mean_t,
        inverse_target_fn=bundle.inverse_target_array,
    )

    pred_target = apply_target_transform(
        y_raw=pred_raw,
        target_mode=bundle.target_mode,
        target_mean=bundle.target_mean,
        target_std=bundle.target_std,
        signed_log_pos_gain=float(bundle.signed_log_pos_gain),
        signed_log_neg_gain=float(bundle.signed_log_neg_gain),
    )
    run_seconds = float(time.perf_counter() - run_t0)

    return TrainingOutput(
        pred_target=pred_target.astype(np.float32),
        pred_raw=pred_raw.astype(np.float32),
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        model_total_params=params["total"],
        model_trainable_params=params["trainable"],
        history=history,
        train_seconds=run_seconds,
    )
