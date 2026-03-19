"""
Visualization module for ConsumerSim results.
"""
import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.agents.base import ConsumerAgent

logger = logging.getLogger(__name__)

QUESTIONS = ["PAGO", "DUR", "PEXP", "BUS12", "BUS5"]
QUESTION_LABELS = {
    "PAGO": "Personal Finances\n(vs Last Year)",
    "DUR": "Durable Goods\nBuying Conditions",
    "PEXP": "Expected Personal\nFinances",
    "BUS12": "Business Outlook\n(12 months)",
    "BUS5": "Business Outlook\n(5 years)",
}
COLORS = {
    "better": "#2ecc71", "good": "#2ecc71",
    "same": "#95a5a6", "uncertain": "#f39c12",
    "worse": "#e74c3c", "bad": "#e74c3c",
}


def plot_response_distribution(
    agents: List[ConsumerAgent],
    output_path: str,
    title: str = "Simulated Survey Response Distribution",
) -> str:
    """Stacked bar chart of response distributions for each question."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    n = len(agents)
    for ax, q in zip(axes, QUESTIONS):
        responses = [getattr(a.response, q) for a in agents]
        cats = list({"PAGO": ["better", "same", "worse"], "DUR": ["good", "uncertain", "bad"],
                     "PEXP": ["better", "same", "worse"], "BUS12": ["good", "uncertain", "bad"],
                     "BUS5": ["good", "uncertain", "bad"]}[q])
        counts = {c: responses.count(c) / n * 100 for c in cats}

        bars = ax.bar(cats, [counts[c] for c in cats],
                      color=[COLORS[c] for c in cats], edgecolor="white", linewidth=0.5)
        for bar, cat in zip(bars, cats):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9)

        ax.set_title(QUESTION_LABELS[q], fontsize=10)
        ax.set_ylabel("% of respondents")
        ax.set_ylim(0, 80)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Response distribution plot saved: {output_path}")
    return output_path


def plot_index_timeseries(
    results_df: pd.DataFrame,
    output_path: str,
    reference: Optional[Dict[int, float]] = None,
) -> str:
    """
    Plot ICS/ICC/ICE time series.
    Optionally overlay reference NBS CCI (rescaled).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"ICS": "#2980b9", "ICC": "#27ae60", "ICE": "#e67e22"}
    for col, color in colors.items():
        if col in results_df.columns:
            ax.plot(results_df.index, results_df[col],
                    marker="o", linewidth=2, label=col, color=color)

    if reference is not None:
        ref_years = [y for y in results_df.index if y in reference]
        if ref_years:
            ref_vals = [reference[y] for y in ref_years]
            # Rescale to match ICS range
            sim_mean = results_df["ICS"].mean()
            ref_mean = np.mean(ref_vals)
            scale = sim_mean / ref_mean if ref_mean != 0 else 1
            ref_rescaled = [v * scale for v in ref_vals]
            ax.plot(ref_years, ref_rescaled, "--", color="gray",
                    linewidth=1.5, label="NBS CCI (rescaled)", alpha=0.7)

    ax.set_xlabel("Year")
    ax.set_ylabel("Index Value")
    ax.set_title("ConsumerSim: Consumer Confidence Indices Over Time", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Index time series plot saved: {output_path}")
    return output_path


def plot_demographic_breakdown(
    breakdowns: Dict[str, pd.DataFrame],
    output_path: str,
) -> str:
    """Heatmap of ICS by demographic group."""
    fig, axes = plt.subplots(1, len(breakdowns), figsize=(4 * len(breakdowns), 5))
    if len(breakdowns) == 1:
        axes = [axes]

    import matplotlib.cm as cm

    for ax, (name, df) in zip(axes, breakdowns.items()):
        if "ICS" not in df.columns:
            continue
        groups = df.index.tolist()
        vals = df["ICS"].values
        bars = ax.barh(groups, vals, color=cm.RdYlGn(np.array(vals) / 150))
        ax.axvline(x=100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("ICS")
        ax.set_title(name.replace("by_", "By ").replace("_", " ").title())
        for bar, val in zip(bars, vals):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9)

    plt.suptitle("ICS by Demographic Group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Demographic breakdown plot saved: {output_path}")
    return output_path


def plot_agent_profile_distribution(
    agents: List[ConsumerAgent],
    output_path: str,
) -> str:
    """Distribution plots of key agent profile attributes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Agent Population Profile Distributions", fontsize=14, fontweight="bold")

    ages = [a.age for a in agents]
    incomes = [a.income_percentile for a in agents]
    happiness = [a.happiness for a in agents]
    confidence = [a.future_confidence for a in agents]

    axes[0, 0].hist(ages, bins=30, color="#3498db", edgecolor="white")
    axes[0, 0].set_title("Age Distribution")
    axes[0, 0].set_xlabel("Age")

    axes[0, 1].hist(incomes, bins=30, color="#2ecc71", edgecolor="white")
    axes[0, 1].set_title("Income Percentile Distribution")
    axes[0, 1].set_xlabel("Income Percentile")

    axes[0, 2].hist(happiness, bins=20, color="#e67e22", edgecolor="white")
    axes[0, 2].set_title("Happiness Distribution (1–10)")
    axes[0, 2].set_xlabel("Happiness Score")

    axes[1, 0].hist(confidence, bins=10, color="#9b59b6", edgecolor="white")
    axes[1, 0].set_title("Future Confidence Distribution (1–5)")
    axes[1, 0].set_xlabel("Confidence Score")

    # Urban/Rural pie
    urban_counts = {"Urban": sum(1 for a in agents if a.urban == "urban"),
                    "Rural": sum(1 for a in agents if a.urban == "rural")}
    axes[1, 1].pie(urban_counts.values(), labels=urban_counts.keys(),
                   autopct="%1.1f%%", colors=["#3498db", "#27ae60"])
    axes[1, 1].set_title("Urban vs Rural")

    # Region distribution
    regions = [a.region for a in agents]
    region_counts = {r: regions.count(r) for r in set(regions)}
    axes[1, 2].bar(region_counts.keys(), region_counts.values(),
                   color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[1, 2].set_title("Regional Distribution")
    axes[1, 2].set_xlabel("Region")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Agent profile plot saved: {output_path}")
    return output_path


def plot_validation_summary(
    micro_df: pd.DataFrame,
    structural_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Summary plot of validation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Micro validation: bar chart of abs_diff per question per response
    micro_pivot = micro_df.pivot(index="response", columns="question", values="abs_diff")
    micro_pivot.plot(kind="bar", ax=axes[0], colormap="Set2")
    axes[0].set_title("Micro Validation: |Simulated - Benchmark| (%)")
    axes[0].set_xlabel("Response Category")
    axes[0].set_ylabel("Absolute Difference (%)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].tick_params(axis="x", rotation=0)

    # Structural validation: pass/fail
    colors_sv = ["#2ecc71" if p else "#e74c3c" for p in structural_df["pass"]]
    y_pos = range(len(structural_df))
    axes[1].barh(y_pos, structural_df["statistic"].abs(), color=colors_sv)
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(structural_df["test"], fontsize=9)
    axes[1].set_title("Structural Validation: Test Statistics")
    axes[1].set_xlabel("|Statistic|")

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(color="#2ecc71", label="Pass"), Patch(color="#e74c3c", label="Fail")]
    axes[1].legend(handles=legend_elems, loc="lower right")

    plt.suptitle("Validation Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Validation summary plot saved: {output_path}")
    return output_path
