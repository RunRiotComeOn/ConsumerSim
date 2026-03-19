"""
US Micro Validation: chi-square test against Michigan Survey benchmark distributions.

Benchmark distributions approximated from published Michigan ICS data (~2023-2024).
"""
import logging
from typing import List

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Approximate benchmark distributions from recent Michigan Survey data
US_BENCHMARK_DISTRIBUTIONS = {
    "PAGO": {"better": 0.28, "same": 0.40, "worse": 0.32},
    "DUR":  {"good": 0.25, "uncertain": 0.38, "bad": 0.37},
    "PEXP": {"better": 0.35, "same": 0.38, "worse": 0.27},
    "BUS12": {"good": 0.22, "uncertain": 0.38, "bad": 0.40},
    "BUS5":  {"good": 0.30, "uncertain": 0.38, "bad": 0.32},
}


def run_micro_validation_us(agents: List) -> pd.DataFrame:
    """Run chi-square goodness-of-fit test for US simulation."""
    n = len(agents)
    results = []

    for q, bench in US_BENCHMARK_DISTRIBUTIONS.items():
        responses = [getattr(a.response, q) for a in agents]
        categories = list(bench.keys())

        observed = [sum(1 for r in responses if r == c) for c in categories]
        expected = [bench[c] * n for c in categories]

        chi2, p_value = stats.chisquare(observed, expected)

        sim_dist = {c: o / n for c, o in zip(categories, observed)}

        results.append({
            "question": q,
            "chi2": round(chi2, 2),
            "p_value": round(p_value, 4),
            "pass": p_value > 0.05,
            "sim_dist": str({k: round(v, 3) for k, v in sim_dist.items()}),
            "bench_dist": str({k: round(v, 3) for k, v in bench.items()}),
        })

        status = "PASS" if p_value > 0.05 else "FAIL"
        logger.info(f"  {q}: chi2={chi2:.2f}, p={p_value:.4f} [{status}]")

    return pd.DataFrame(results)
