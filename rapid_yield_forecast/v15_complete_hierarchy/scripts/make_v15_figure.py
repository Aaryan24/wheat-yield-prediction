#!/usr/bin/env python3
"""Make the compact V15 result figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


V15 = Path(__file__).resolve().parents[1]
A = V15 / "artifacts"


def main() -> None:
    metrics = pd.read_csv(A / "point_model_ablation_metrics.csv")
    audit = pd.read_csv(A / "point_year_state_audit.csv")
    distribution = pd.read_csv(A / "v15_distribution_metrics.csv")
    labels = {
        "V5_production": "V5",
        "V14_future_weather_shadow": "V14 weather",
        "V15_combined_regularized": "V15 combined",
    }
    colors = {
        "V5_production": "#64748b",
        "V14_future_weather_shadow": "#0ea5e9",
        "V15_combined_regularized": "#16a34a",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    periods = ["development", "late", "four_year"]
    x = np.arange(len(periods))
    width = 0.24
    for offset, model in enumerate(labels):
        block = metrics[
            metrics["model"].eq(model) & metrics["period"].isin(periods)
        ].set_index("period").loc[periods]
        axes[0].bar(
            x + (offset - 1) * width, block["rmse"], width,
            label=labels[model], color=colors[model],
        )
    axes[0].set_xticks(x, ["2019–20\nselection", "2021–22\nuntouched", "2019–22\nall"])
    axes[0].set_ylabel("RMSE (kg/ha; lower is better)")
    axes[0].set_title("Point forecast")
    axes[0].legend(frameon=False, fontsize=8)

    year = audit[
        audit["level"].eq("year") & audit["model"].isin(labels)
    ]
    for model in labels:
        block = year[year["model"].eq(model)].sort_values("season_start_year")
        axes[1].plot(
            block["season_start_year"], block["rmse"],
            marker="o", linewidth=2, label=labels[model], color=colors[model],
        )
    axes[1].set_xticks([2019, 2020, 2021, 2022])
    axes[1].set_ylabel("RMSE (kg/ha)")
    axes[1].set_title("Year-by-year honesty check")

    dist = distribution[
        distribution["period"].isin(["development", "late", "four_year"])
    ].set_index("period").loc[periods]
    axes[2].bar(x - width / 2, dist["coverage_80"] * 100, width, label="80% range")
    axes[2].bar(x + width / 2, dist["coverage_90"] * 100, width, label="90% range")
    axes[2].axhline(80, color="#2563eb", linestyle="--", linewidth=1)
    axes[2].axhline(90, color="#f97316", linestyle="--", linewidth=1)
    axes[2].set_ylim(60, 100)
    axes[2].set_xticks(x, ["2019–20", "2021–22", "2019–22"])
    axes[2].set_ylabel("Outcomes inside range (%)")
    axes[2].set_title("Probability-range calibration")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "V15: future weather + current crop condition",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(A / "v15_result_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
