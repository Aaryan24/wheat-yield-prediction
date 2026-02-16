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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from codex_v2.src.data.build_dataset_v2 import DatasetBundle
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
    )


def _warmup_cosine_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _predict_indices(
    model: torch.nn.Module,
    idx: np.ndarray,
    weather_t: torch.Tensor,
    weather_mask_t: torch.Tensor,
    sat_t: torch.Tensor,
    sat_mask_t: torch.Tensor,
    y_target_t: torch.Tensor,
    opdate_idx_t: torch.Tensor,
    adj_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if len(idx) == 0:
        return np.zeros((0, y_target_t.shape[1]), dtype=np.float32), np.zeros((0, y_target_t.shape[1]), dtype=np.float32)

    i_t = torch.tensor(idx, dtype=torch.long, device=weather_t.device)
    with torch.no_grad():
        pred_t, _ = model(
            weather_t[i_t],
            weather_mask_t[i_t],
            sat_t[i_t],
            sat_mask_t[i_t],
            adj_t,
            opdate_idx=opdate_idx_t[i_t],
        )
    y_true = y_target_t[i_t].detach().cpu().numpy().astype(np.float32)
    y_pred = pred_t.detach().cpu().numpy().astype(np.float32)
    return y_true, y_pred


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
    y_target_t = torch.tensor(bundle.y_target, dtype=torch.float32, device=device)
    opdate_idx_t = torch.tensor(bundle.sample_opdate_idx, dtype=torch.long, device=device)
    adj_t = torch.tensor(bundle.adjacency, dtype=torch.float32, device=device)

    criterion = build_loss(str(optimization.get("loss", "huber")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        IndexDataset(bundle.train_idx),
        batch_size=batch_size,
        shuffle=True,
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
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0
    step = 0
    history: List[Dict[str, float]] = []

    run_t0 = time.perf_counter()

    pbar = tqdm(range(1, epochs + 1), desc=f"train[{mode}|{target_mode}|h{horizon_days}]", leave=True)
    with training_log_path.open("w") as log_fh:
        for epoch in pbar:
            model.train()
            train_losses: List[float] = []

            for idx_batch in train_loader:
                b = idx_batch.to(device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                pred, _ = model(
                    weather_t[b],
                    weather_mask_t[b],
                    sat_t[b],
                    sat_mask_t[b],
                    adj_t,
                    opdate_idx=opdate_idx_t[b],
                )
                loss = criterion(pred, y_target_t[b])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()

                train_losses.append(float(loss.item()))
                step += 1

            model.eval()
            val_true_t, val_pred_t = _predict_indices(
                model=model,
                idx=bundle.val_idx,
                weather_t=weather_t,
                weather_mask_t=weather_mask_t,
                sat_t=sat_t,
                sat_mask_t=sat_mask_t,
                y_target_t=y_target_t,
                opdate_idx_t=opdate_idx_t,
                adj_t=adj_t,
            )
            val_true_raw = bundle.inverse_target_array(val_true_t)
            val_pred_raw = bundle.inverse_target_array(val_pred_t)
            val_metrics = regression_metrics(val_true_raw.ravel(), val_pred_raw.ravel())

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            row = {
                "epoch": int(epoch),
                "seed": int(seed),
                "train_loss": train_loss,
                "val_rmse": float(val_metrics["rmse"]),
                "val_mae": float(val_metrics["mae"]),
                "val_r2": float(val_metrics["r2"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()

            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_rmse=f"{val_metrics['rmse']:.2f}",
                val_r2=f"{val_metrics['r2']:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if val_metrics["rmse"] < best_val_rmse - 1e-8:
                best_val_rmse = float(val_metrics["rmse"])
                best_epoch = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
                torch.save(best_state, model_ckpt)
            else:
                wait += 1
                if wait >= patience:
                    break

    epochs_ran = len(history)
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, model_ckpt)
    model.load_state_dict(best_state)
    model.eval()

    all_idx = np.arange(bundle.y_target.shape[0], dtype=np.int64)
    _, pred_target = _predict_indices(
        model=model,
        idx=all_idx,
        weather_t=weather_t,
        weather_mask_t=weather_mask_t,
        sat_t=sat_t,
        sat_mask_t=sat_mask_t,
        y_target_t=y_target_t,
        opdate_idx_t=opdate_idx_t,
        adj_t=adj_t,
    )

    pred_raw = bundle.inverse_target_array(pred_target)
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
