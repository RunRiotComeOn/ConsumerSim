"""
US-specific visualization module.
"""
import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_us_agent_profiles(agents: List, output_path: str) -> str:
    """Distribution plots of US agent profile attributes."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("US Agent Population Profile Distributions", fontsize=14, fontweight="bold")

    ages = [a.age for a in agents]
    incomes = [a.income_percentile for a in agents]
    happiness = [a.happiness for a in agents]
    confidence = [a.future_confidence for a in agents]

    axes[0, 0].hist(ages, bins=30, color="#3498db", edgecolor="white")
    axes[0, 0].set_title("Age Distribution")

    axes[0, 1].hist(incomes, bins=30, color="#2ecc71", edgecolor="white")
    axes[0, 1].set_title("Income Percentile")

    axes[0, 2].hist(happiness, bins=10, color="#e67e22", edgecolor="white")
    axes[0, 2].set_title("Happiness (1-3, GSS)")

    axes[0, 3].hist(confidence, bins=10, color="#9b59b6", edgecolor="white")
    axes[0, 3].set_title("Future Confidence (1-5)")

    # Race pie
    race_counts = {}
    for a in agents:
        race_counts[a.race] = race_counts.get(a.race, 0) + 1
    axes[1, 0].pie(race_counts.values(), labels=race_counts.keys(),
                   autopct="%1.1f%%", colors=["#3498db", "#e74c3c", "#f39c12", "#2ecc71", "#95a5a6"])
    axes[1, 0].set_title("Race/Ethnicity")

    # Party pie
    party_counts = {}
    for a in agents:
        party_counts[a.political_leaning] = party_counts.get(a.political_leaning, 0) + 1
    axes[1, 1].pie(party_counts.values(), labels=party_counts.keys(),
                   autopct="%1.1f%%", colors=["#2980b9", "#95a5a6", "#c0392b"])
    axes[1, 1].set_title("Political Leaning")

    # Metro/Non-metro pie
    metro_counts = {"Metro": sum(1 for a in agents if a.metro_status == "metro"),
                    "Non-metro": sum(1 for a in agents if a.metro_status == "non-metro")}
    axes[1, 2].pie(metro_counts.values(), labels=metro_counts.keys(),
                   autopct="%1.1f%%", colors=["#3498db", "#27ae60"])
    axes[1, 2].set_title("Metro Status")

    # Region bar
    regions = [a.region for a in agents]
    region_counts = {r: regions.count(r) for r in sorted(set(regions))}
    axes[1, 3].bar(region_counts.keys(), region_counts.values(),
                   color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"])
    axes[1, 3].set_title("Census Region")
    axes[1, 3].tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"US agent profile plot saved: {output_path}")
    return output_path


def plot_us_validation_summary(micro_df: pd.DataFrame, structural_df: pd.DataFrame,
                                output_path: str) -> str:
    """US validation summary plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Micro: chi-square by question
    questions = micro_df["question"].tolist()
    chi2_vals = micro_df["chi2"].tolist()
    pass_flags = micro_df["pass"].tolist()
    colors_m = ["#2ecc71" if p else "#e74c3c" for p in pass_flags]
    axes[0].bar(questions, chi2_vals, color=colors_m)
    axes[0].set_title("Micro Validation: Chi-square Statistics")
    axes[0].set_ylabel("Chi-square value")
    axes[0].axhline(y=5.99, color="orange", linestyle="--", label="Critical (df=2, p=0.05)")
    axes[0].legend()

    # Structural: pass/fail
    tests = structural_df["test"].tolist()
    pass_s = structural_df["pass"].tolist()
    colors_s = ["#2ecc71" if p else "#e74c3c" for p in pass_s]
    y_pos = range(len(tests))
    axes[1].barh(y_pos, [1] * len(tests), color=colors_s)
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(tests, fontsize=8)
    axes[1].set_title("Structural Validation: Pass/Fail")
    axes[1].set_xlim(0, 1.5)
    axes[1].set_xticks([])
    for i, (test, passed) in enumerate(zip(tests, pass_s)):
        axes[1].text(1.05, i, "PASS" if passed else "FAIL",
                     va="center", fontsize=9, fontweight="bold",
                     color="#2ecc71" if passed else "#e74c3c")

    from matplotlib.patches import Patch
    legend_elems = [Patch(color="#2ecc71", label="Pass"), Patch(color="#e74c3c", label="Fail")]
    axes[1].legend(handles=legend_elems, loc="lower right")

    plt.suptitle("US Simulation Validation Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"US validation summary plot saved: {output_path}")
    return output_path
