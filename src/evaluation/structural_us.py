"""
US Structural Validation: tests that simulation reproduces known US economic relationships.

Key US-specific relationships to validate:
1. Income -> Sentiment (positive correlation)
2. Metro > Non-metro (metro residents more optimistic)
3. White > Black sentiment gap (structural inequality)
4. Republican vs Democrat gap (partisan perception)
5. Homeowners vs Renters
6. Young > Old on expectations
7. Education -> Sentiment
"""
import logging
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _sentiment_score(agent) -> float:
    """Combined sentiment score from 5 survey responses."""
    score = 0
    r = agent.response
    # PAGO / PEXP
    for val in [r.PAGO, r.PEXP]:
        if val == "better": score += 1
        elif val == "worse": score -= 1
    # DUR / BUS12 / BUS5
    for val in [r.DUR, r.BUS12, r.BUS5]:
        if val == "good": score += 1
        elif val == "bad": score -= 1
    return score


def run_structural_validation_us(agents: List) -> pd.DataFrame:
    """Validate 7 structural economic relationships for US simulation."""
    logger.info("Running US structural validation...")

    scores = [_sentiment_score(a) for a in agents]
    results = []

    # 1. Income -> Sentiment (Pearson r)
    incomes = [a.income_percentile for a in agents]
    r, p = stats.pearsonr(incomes, scores)
    results.append({
        "test": "Income -> Sentiment",
        "statistic": f"r={r:.3f}",
        "p_value": round(p, 4),
        "expected": "r > 0",
        "pass": r > 0 and p < 0.05,
    })
    logger.info(f"  Income->Sentiment: r={r:.3f}, p={p:.4f}")

    # 2. Metro > Non-metro
    metro_scores = [s for a, s in zip(agents, scores) if a.metro_status == "metro"]
    nonmetro_scores = [s for a, s in zip(agents, scores) if a.metro_status == "non-metro"]
    if metro_scores and nonmetro_scores:
        t, p = stats.ttest_ind(metro_scores, nonmetro_scores)
        results.append({
            "test": "Metro > Non-metro",
            "statistic": f"t={t:.3f}",
            "p_value": round(p, 4),
            "expected": "t > 0",
            "pass": t > 0,
        })
        logger.info(f"  Metro>NonMetro: t={t:.3f}, p={p:.4f}")

    # 3. White-Black gap
    white_scores = [s for a, s in zip(agents, scores) if a.race == "White"]
    black_scores = [s for a, s in zip(agents, scores) if a.race == "Black"]
    if white_scores and black_scores:
        t, p = stats.ttest_ind(white_scores, black_scores)
        w_mean = np.mean(white_scores)
        b_mean = np.mean(black_scores)
        results.append({
            "test": "White > Black sentiment",
            "statistic": f"White={w_mean:.2f}, Black={b_mean:.2f}, t={t:.3f}",
            "p_value": round(p, 4),
            "expected": "White > Black",
            "pass": w_mean > b_mean,
        })
        logger.info(f"  White({w_mean:.2f}) vs Black({b_mean:.2f}): t={t:.3f}, p={p:.4f}")

    # 4. Republican vs Democrat gap (R more optimistic under neutral scenario)
    rep_scores = [s for a, s in zip(agents, scores) if a.political_leaning == "Republican"]
    dem_scores = [s for a, s in zip(agents, scores) if a.political_leaning == "Democrat"]
    if rep_scores and dem_scores:
        t, p = stats.ttest_ind(rep_scores, dem_scores)
        r_mean = np.mean(rep_scores)
        d_mean = np.mean(dem_scores)
        results.append({
            "test": "Partisan gap exists",
            "statistic": f"Rep={r_mean:.2f}, Dem={d_mean:.2f}, t={t:.3f}",
            "p_value": round(p, 4),
            "expected": "Gap exists (|t| > 0)",
            "pass": abs(t) > 1.0,
        })
        logger.info(f"  Rep({r_mean:.2f}) vs Dem({d_mean:.2f}): t={t:.3f}, p={p:.4f}")

    # 5. Homeowners vs Renters
    owner_scores = [s for a, s in zip(agents, scores) if a.homeownership == "owner"]
    renter_scores = [s for a, s in zip(agents, scores) if a.homeownership == "renter"]
    if owner_scores and renter_scores:
        t, p = stats.ttest_ind(owner_scores, renter_scores)
        results.append({
            "test": "Homeowner > Renter",
            "statistic": f"t={t:.3f}",
            "p_value": round(p, 4),
            "expected": "t > 0",
            "pass": t > 0,
        })
        logger.info(f"  Owner>Renter: t={t:.3f}, p={p:.4f}")

    # 6. Young > Old on expectations
    young_exp = []
    old_exp = []
    for a in agents:
        exp_score = 0
        if a.response.PEXP == "better": exp_score += 1
        elif a.response.PEXP == "worse": exp_score -= 1
        if a.response.BUS5 == "good": exp_score += 1
        elif a.response.BUS5 == "bad": exp_score -= 1
        if a.age < 35:
            young_exp.append(exp_score)
        elif a.age >= 55:
            old_exp.append(exp_score)
    if young_exp and old_exp:
        t, p = stats.ttest_ind(young_exp, old_exp)
        results.append({
            "test": "Young > Old (expectations)",
            "statistic": f"t={t:.3f}",
            "p_value": round(p, 4),
            "expected": "t > 0",
            "pass": t > 0,
        })
        logger.info(f"  Young>Old expectations: t={t:.3f}, p={p:.4f}")

    # 7. Education -> Sentiment
    edu_order = {"less_than_hs": 1, "high_school": 2, "some_college": 3,
                 "bachelors": 4, "graduate": 5}
    edu_nums = [edu_order.get(a.education, 3) for a in agents]
    r, p = stats.pearsonr(edu_nums, scores)
    results.append({
        "test": "Education -> Sentiment",
        "statistic": f"r={r:.3f}",
        "p_value": round(p, 4),
        "expected": "r > 0",
        "pass": r > 0 and p < 0.05,
    })
    logger.info(f"  Education->Sentiment: r={r:.3f}, p={p:.4f}")

    return pd.DataFrame(results)
