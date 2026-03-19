"""
Index computation: ICC, ICE, ICS from aggregated survey responses.

Formula (Michigan Consumer Sentiment Index style):
  Xi = (%positive - %negative) + 100

Where:
  PAGO_R:  positive="better",  negative="worse"
  DUR_R:   positive="good",    negative="bad"
  PEXP_R:  positive="better",  negative="worse"
  BUS12_R: positive="good",    negative="bad"
  BUS5_R:  positive="good",    negative="bad"

  ICC = mean(PAGO_R, DUR_R)
  ICE = mean(PEXP_R, BUS12_R, BUS5_R)
  ICS = mean(PAGO_R, DUR_R, PEXP_R, BUS12_R, BUS5_R)
"""
import logging
from typing import Dict, List

import pandas as pd

from src.agents.base import ConsumerAgent

logger = logging.getLogger(__name__)

POSITIVE_MAP = {
    "PAGO": "better",
    "DUR": "good",
    "PEXP": "better",
    "BUS12": "good",
    "BUS5": "good",
}
NEGATIVE_MAP = {
    "PAGO": "worse",
    "DUR": "bad",
    "PEXP": "worse",
    "BUS12": "bad",
    "BUS5": "bad",
}


def compute_relative_index(responses: List[str], positive: str, negative: str) -> float:
    """Compute Xi = (%pos - %neg) + 100 for one question."""
    n = len(responses)
    if n == 0:
        return 100.0
    pct_pos = sum(r == positive for r in responses) / n * 100
    pct_neg = sum(r == negative for r in responses) / n * 100
    return pct_pos - pct_neg + 100.0


def compute_indices(agents: List[ConsumerAgent]) -> Dict[str, float]:
    """
    Compute all relative indices and composite ICC/ICE/ICS.

    Returns dict with keys:
      PAGO_R, DUR_R, PEXP_R, BUS12_R, BUS5_R, ICC, ICE, ICS
    """
    questions = ["PAGO", "DUR", "PEXP", "BUS12", "BUS5"]
    response_lists = {q: [] for q in questions}

    for agent in agents:
        response_lists["PAGO"].append(agent.response.PAGO)
        response_lists["DUR"].append(agent.response.DUR)
        response_lists["PEXP"].append(agent.response.PEXP)
        response_lists["BUS12"].append(agent.response.BUS12)
        response_lists["BUS5"].append(agent.response.BUS5)

    indices = {}
    for q in questions:
        xi = compute_relative_index(
            response_lists[q], POSITIVE_MAP[q], NEGATIVE_MAP[q]
        )
        indices[f"{q}_R"] = round(xi, 2)

    indices["ICC"] = round((indices["PAGO_R"] + indices["DUR_R"]) / 2, 2)
    indices["ICE"] = round((indices["PEXP_R"] + indices["BUS12_R"] + indices["BUS5_R"]) / 3, 2)
    indices["ICS"] = round(sum(indices[f"{q}_R"] for q in questions) / 5, 2)

    return indices


def compute_indices_by_group(agents: List[ConsumerAgent], group_by: str) -> pd.DataFrame:
    """
    Compute ICS broken down by a demographic attribute.
    group_by: "region", "urban", "age_group", "education", "income_tercile"
    """
    rows = []
    for agent in agents:
        if group_by == "age_group":
            if agent.age < 35:
                group_val = "18-34"
            elif agent.age < 55:
                group_val = "35-54"
            else:
                group_val = "55+"
        elif group_by == "income_tercile":
            if agent.income_percentile < 33:
                group_val = "low"
            elif agent.income_percentile < 67:
                group_val = "middle"
            else:
                group_val = "high"
        else:
            group_val = getattr(agent, group_by, "unknown")

        rows.append({
            "group": group_val,
            "PAGO": agent.response.PAGO,
            "DUR": agent.response.DUR,
            "PEXP": agent.response.PEXP,
            "BUS12": agent.response.BUS12,
            "BUS5": agent.response.BUS5,
        })

    df = pd.DataFrame(rows)
    results = []
    for group_val, sub in df.groupby("group"):
        sub_agents = [
            type("_A", (), {
                "response": type("_R", (), {
                    "PAGO": row["PAGO"], "DUR": row["DUR"],
                    "PEXP": row["PEXP"], "BUS12": row["BUS12"], "BUS5": row["BUS5"],
                })()
            })()
            for _, row in sub.iterrows()
        ]
        idx = compute_indices(sub_agents)
        idx["group"] = group_val
        idx["n"] = len(sub)
        results.append(idx)

    return pd.DataFrame(results).set_index("group")
