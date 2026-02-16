#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Allow running from repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _find_top_level_runs(out_root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for cfg_path in out_root.rglob("model_config_resolved.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if isinstance(cfg, dict) and "runs" in cfg:
            run_dirs.append(cfg_path.parent)
    return sorted(set(run_dirs))


def _ablation_id_from_run_name(run_name: str) -> str:
    token = str(run_name).split("_")[0].strip()
    if token in {"A0", "A1", "A2", "B0", "B1", "B2", "B3", "B4"}:
        return token
    return "custom"


def _df_to_markdown_simple(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows_"

    head = df.head(max_rows).copy()
    cols = [str(c) for c in head.columns.tolist()]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in head.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"_... showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def compile_all_runs(out_root: Path, write_report: bool = True) -> Dict[str, Path]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[pd.DataFrame] = []
    run_dirs = _find_top_level_runs(out_root)

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics_per_opdate.csv"
        cfg_path = run_dir / "model_config_resolved.json"
        if not metrics_path.exists() or not cfg_path.exists():
            continue

        mdf = pd.read_csv(metrics_path)
        cfg = json.loads(cfg_path.read_text())
        mdf["run_dir"] = str(run_dir)
        mdf["run_name"] = str(cfg.get("run_name", run_dir.name))
        mdf["horizon_days"] = int(cfg.get("horizon_days", -1))
        mdf["seed"] = int(cfg.get("seed", -1))
        mdf["mode"] = str(cfg.get("mode", "unknown"))
        mdf["target_mode"] = str(cfg.get("target_mode", "unknown"))
        mdf["ablation_id"] = _ablation_id_from_run_name(str(cfg.get("run_name", run_dir.name)))
        rows.append(mdf)

    if rows:
        compiled = pd.concat(rows, ignore_index=True)
    else:
        compiled = pd.DataFrame(
            columns=[
                "run_name",
                "ablation_id",
                "mode",
                "target_mode",
                "horizon_days",
                "seed",
                "split",
                "operational_date",
                "rmse",
                "mae",
                "mape",
                "r2",
            ]
        )

    csv_path = out_root / "v2_all_runs_compiled.csv"
    compiled.to_csv(csv_path, index=False)

    md_path = out_root / "v2_all_runs_report.md"
    if write_report:
        lines: List[str] = ["# Codex V2 Run Report", ""]
        lines.append(f"- Compiled runs: **{compiled['run_name'].nunique() if not compiled.empty else 0}**")
        lines.append(f"- Total metric rows: **{len(compiled)}**")
        lines.append("")

        if not compiled.empty:
            test_df = compiled[compiled["split"] == "test"].copy()
            if not test_df.empty:
                run_level = (
                    test_df.groupby(["run_name", "ablation_id", "mode", "target_mode", "horizon_days", "seed"], dropna=False)
                    .agg(
                        test_rmse_mean=("rmse", "mean"),
                        test_mae_mean=("mae", "mean"),
                        test_r2_mean=("r2", "mean"),
                        n_opdates=("operational_date", "nunique"),
                    )
                    .reset_index()
                )
                seeded = (
                    run_level.groupby(["ablation_id", "mode", "target_mode", "horizon_days"], dropna=False)
                    .agg(
                        seeds=("seed", "nunique"),
                        mean_test_rmse=("test_rmse_mean", "mean"),
                        std_test_rmse=("test_rmse_mean", "std"),
                        mean_test_r2=("test_r2_mean", "mean"),
                        std_test_r2=("test_r2_mean", "std"),
                    )
                    .reset_index()
                    .sort_values(["horizon_days", "mean_test_rmse"], ascending=[True, True])
                )

                lines.append("## Best Runs (Test RMSE)")
                lines.append("")
                top_rmse = run_level.sort_values("test_rmse_mean").head(10)
                lines.append(_df_to_markdown_simple(top_rmse, max_rows=10))
                lines.append("")

                lines.append("## Seeded Summary")
                lines.append("")
                lines.append(_df_to_markdown_simple(seeded, max_rows=20))
                lines.append("")

                positive_r2_share = (
                    test_df.assign(_r2_pos=(test_df["r2"] > 0).astype(float))
                    .groupby(["run_name"], dropna=False)["_r2_pos"]
                    .mean()
                    .mul(100.0)
                    .reset_index(name="pct_opdates_r2_gt_0")
                    .sort_values("pct_opdates_r2_gt_0", ascending=False)
                )
                lines.append("## Positive R2 Share by Run")
                lines.append("")
                lines.append(_df_to_markdown_simple(positive_r2_share.head(10), max_rows=10))
                lines.append("")

        md_path.write_text("\n".join(lines))

    return {"csv_path": csv_path, "md_path": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Codex V2 run reports.")
    parser.add_argument("--out-root", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = compile_all_runs(out_root=Path(args.out_root), write_report=True)
    print(json.dumps({"compiled_csv": str(out["csv_path"]), "compiled_md": str(out["md_path"])}, indent=2))


if __name__ == "__main__":
    main()
