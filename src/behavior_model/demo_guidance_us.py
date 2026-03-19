"""
US Demographic Guidance for Behavior Prediction.

72-group prior distributions stratified by:
  metro/non-metro (2) x age (3) x income (3) x race (4) = 72 groups
Plus party shift as an additive correction (not cross-product).

Priors are dynamically calibrated to the latest real Michigan ICS (UMCSENT)
so the baseline matches the current sentiment regime. Demographic adjustments
(race, income, age, metro) and party shifts are applied on top.
"""
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

DemoGroupKey = Tuple[str, str, str, str]  # (metro, age_group, income_tier, race)

QUESTIONS = ["PAGO", "DUR", "PEXP", "BUS12", "BUS5"]


def _agent_demo_group(agent) -> DemoGroupKey:
    """Classify a US agent into a demographic group key."""
    metro = "metro" if agent.metro_status == "metro" else "non-metro"

    if agent.age < 35:
        ag = "young"
    elif agent.age < 55:
        ag = "middle"
    else:
        ag = "elderly"

    if agent.income_percentile < 33.33:
        it = "low"
    elif agent.income_percentile < 66.67:
        it = "mid"
    else:
        it = "high"

    race = agent.race if agent.race in ("White", "Black", "Hispanic", "Asian") else "Other"

    return (metro, ag, it, race)


# ---------------------------------------------------------------------------
# Party shift: additive correction to positive/negative probabilities
# This captures the ~30-40 point partisan gap in US consumer sentiment
# Party effect is modeled as independent of demographic cross-product
# ---------------------------------------------------------------------------
PARTY_SHIFT = {
    # Under a generic/neutral scenario (no strong partisan lean in economy)
    # These shifts represent the baseline partisan perception gap
    "Democrat": {
        "PAGO":  (-0.02, +0.02),
        "DUR":   (-0.01, +0.01),
        "PEXP":  (-0.02, +0.02),
        "BUS12": (-0.05, +0.05),  # Business conditions most affected by party
        "BUS5":  (-0.05, +0.05),
    },
    "Independent": {
        "PAGO":  (0.0, 0.0),
        "DUR":   (0.0, 0.0),
        "PEXP":  (0.0, 0.0),
        "BUS12": (0.0, 0.0),
        "BUS5":  (0.0, 0.0),
    },
    "Republican": {
        "PAGO":  (+0.02, -0.02),
        "DUR":   (+0.01, -0.01),
        "PEXP":  (+0.02, -0.02),
        "BUS12": (+0.05, -0.05),
        "BUS5":  (+0.05, -0.05),
    },
}


def _base_ics_from_priors(base: dict) -> float:
    """Compute the implied ICS from base prior distributions."""
    pos_neg = {
        "PAGO": ("better", "worse"), "DUR": ("good", "bad"),
        "PEXP": ("better", "worse"), "BUS12": ("good", "bad"), "BUS5": ("good", "bad"),
    }
    gaps = []
    for q in QUESTIONS:
        pos_label, neg_label = pos_neg[q]
        gaps.append(base[q][pos_label] - base[q][neg_label])
    mean_gap = sum(gaps) / len(gaps)
    return mean_gap * 100 + 100


# "Neutral era" base distributions (ICS ≈ 96.4)
# These are the reference distributions that get shifted to match real ICS
_NEUTRAL_BASE = {
    "PAGO":  {"better": 0.28, "same": 0.42, "worse": 0.30},
    "DUR":   {"good": 0.25, "uncertain": 0.40, "bad": 0.35},
    "PEXP":  {"better": 0.35, "same": 0.40, "worse": 0.25},
    "BUS12": {"good": 0.22, "uncertain": 0.40, "bad": 0.38},
    "BUS5":  {"good": 0.30, "uncertain": 0.40, "bad": 0.30},
}
_NEUTRAL_ICS = _base_ics_from_priors(_NEUTRAL_BASE)


def _shift_base_to_target(target_ics: float) -> dict:
    """Shift base distributions so their implied ICS matches target_ics.

    Uses uniform additive shift across all 5 questions:
      - decrease positive probability by delta
      - increase negative probability by delta
      - neutral absorbs remainder
    """
    target_mean_gap = (target_ics - 100) / 100
    current_mean_gap = (_NEUTRAL_ICS - 100) / 100
    shift = target_mean_gap - current_mean_gap  # negative means more pessimistic
    delta = shift / 2  # split between pos decrease and neg increase

    pos_neg = {
        "PAGO": ("better", "worse"), "DUR": ("good", "bad"),
        "PEXP": ("better", "worse"), "BUS12": ("good", "bad"), "BUS5": ("good", "bad"),
    }
    neutral = {
        "PAGO": "same", "DUR": "uncertain",
        "PEXP": "same", "BUS12": "uncertain", "BUS5": "uncertain",
    }

    shifted = {}
    for q in QUESTIONS:
        pos_label, neg_label = pos_neg[q]
        neu_label = neutral[q]
        p_pos = _NEUTRAL_BASE[q][pos_label] + delta  # delta is negative when pessimistic
        p_neg = _NEUTRAL_BASE[q][neg_label] - delta
        p_pos = max(0.03, p_pos)
        p_neg = max(0.03, p_neg)
        p_neu = max(0.03, 1.0 - p_pos - p_neg)
        total = p_pos + p_neg + p_neu
        shifted[q] = {
            pos_label: p_pos / total,
            neu_label: p_neu / total,
            neg_label: p_neg / total,
        }

    return shifted


def _build_default_priors(target_ics: float = None) -> Dict[DemoGroupKey, Dict[str, Dict[str, float]]]:
    """Build the full 72-group prior table.

    If target_ics is provided, shifts base distributions to match that ICS level.
    Otherwise uses the neutral-era base (~ICS 96).
    """

    if target_ics is not None:
        base = _shift_base_to_target(target_ics)
        implied = _base_ics_from_priors(base)
        logger.info(f"Prior calibrated to target ICS={target_ics:.1f} (implied base ICS={implied:.1f})")
    else:
        base = dict(_NEUTRAL_BASE)

    # --- Adjustment factors ---
    metro_adj = {
        "metro": {"PAGO": (+0.02, -0.01), "DUR": (+0.02, -0.01), "PEXP": (+0.02, -0.01),
                  "BUS12": (+0.01, -0.01), "BUS5": (+0.01, -0.01)},
        "non-metro": {"PAGO": (-0.03, +0.02), "DUR": (-0.03, +0.02), "PEXP": (-0.02, +0.02),
                      "BUS12": (-0.02, +0.01), "BUS5": (-0.02, +0.01)},
    }

    age_adj = {
        "young": {"PAGO": (+0.03, -0.02), "DUR": (+0.01, -0.01), "PEXP": (+0.06, -0.03),
                  "BUS12": (+0.02, -0.01), "BUS5": (+0.04, -0.02)},
        "middle": {"PAGO": (0.0, 0.0), "DUR": (-0.02, +0.02), "PEXP": (0.0, 0.0),
                   "BUS12": (0.0, 0.0), "BUS5": (0.0, 0.0)},
        "elderly": {"PAGO": (-0.04, +0.03), "DUR": (-0.01, +0.01), "PEXP": (-0.05, +0.04),
                    "BUS12": (-0.02, +0.02), "BUS5": (-0.03, +0.02)},
    }

    inc_adj = {
        "low": {"PAGO": (-0.10, +0.08), "DUR": (-0.06, +0.05), "PEXP": (-0.07, +0.06),
                "BUS12": (-0.05, +0.04), "BUS5": (-0.04, +0.03)},
        "mid": {"PAGO": (0.0, 0.0), "DUR": (0.0, 0.0), "PEXP": (0.0, 0.0),
                "BUS12": (0.0, 0.0), "BUS5": (0.0, 0.0)},
        "high": {"PAGO": (+0.12, -0.08), "DUR": (+0.08, -0.05), "PEXP": (+0.10, -0.06),
                 "BUS12": (+0.06, -0.04), "BUS5": (+0.05, -0.03)},
    }

    # Race adjustment: captures the structural Black-White sentiment gap
    # and Hispanic/Asian patterns from published ICS breakdowns
    race_adj = {
        "White": {"PAGO": (+0.04, -0.03), "DUR": (+0.03, -0.02), "PEXP": (+0.03, -0.02),
                  "BUS12": (+0.03, -0.02), "BUS5": (+0.03, -0.02)},
        "Black": {"PAGO": (-0.08, +0.07), "DUR": (-0.06, +0.05), "PEXP": (-0.06, +0.05),
                  "BUS12": (-0.06, +0.05), "BUS5": (-0.05, +0.04)},
        "Hispanic": {"PAGO": (-0.03, +0.03), "DUR": (-0.02, +0.02), "PEXP": (+0.01, -0.01),
                     "BUS12": (-0.03, +0.02), "BUS5": (-0.02, +0.01)},
        "Asian": {"PAGO": (+0.02, -0.01), "DUR": (+0.02, -0.01), "PEXP": (+0.02, -0.01),
                  "BUS12": (+0.01, -0.01), "BUS5": (+0.01, -0.01)},
        "Other": {"PAGO": (0.0, 0.0), "DUR": (0.0, 0.0), "PEXP": (0.0, 0.0),
                  "BUS12": (0.0, 0.0), "BUS5": (0.0, 0.0)},
    }

    pos_neg = {
        "PAGO": ("better", "worse"), "DUR": ("good", "bad"),
        "PEXP": ("better", "worse"), "BUS12": ("good", "bad"), "BUS5": ("good", "bad"),
    }
    neutral = {
        "PAGO": "same", "DUR": "uncertain",
        "PEXP": "same", "BUS12": "uncertain", "BUS5": "uncertain",
    }

    priors: Dict[DemoGroupKey, Dict[str, Dict[str, float]]] = {}

    for metro in ("metro", "non-metro"):
        for ag in ("young", "middle", "elderly"):
            for it in ("low", "mid", "high"):
                for race in ("White", "Black", "Hispanic", "Asian"):
                    key = (metro, ag, it, race)
                    group_dist: Dict[str, Dict[str, float]] = {}

                    for q in QUESTIONS:
                        pos_label, neg_label = pos_neg[q]
                        neu_label = neutral[q]

                        p_pos = base[q][pos_label]
                        p_neg = base[q][neg_label]

                        for adj in (metro_adj[metro], age_adj[ag],
                                    inc_adj[it], race_adj[race]):
                            dp, dn = adj[q]
                            p_pos += dp
                            p_neg += dn

                        p_pos = max(0.02, min(0.96, p_pos))
                        p_neg = max(0.02, min(0.96, p_neg))
                        p_neu = max(0.02, 1.0 - p_pos - p_neg)

                        total = p_pos + p_neg + p_neu
                        group_dist[q] = {
                            pos_label: p_pos / total,
                            neg_label: p_neg / total,
                            neu_label: p_neu / total,
                        }

                    priors[key] = group_dist

    return priors


DEFAULT_DEMO_PRIORS = _build_default_priors()


def calibrate_priors(target_ics: float) -> None:
    """Recalibrate all 72-group priors to match a target ICS level.

    Call this before simulation to anchor priors to the current sentiment regime.
    Typically use the previous month's real UMCSENT as target.
    """
    global DEFAULT_DEMO_PRIORS
    DEFAULT_DEMO_PRIORS = _build_default_priors(target_ics=target_ics)
    logger.info(f"Priors recalibrated to ICS={target_ics:.1f} "
                f"({len(DEFAULT_DEMO_PRIORS)} groups)")


def get_demo_prior(agent) -> Dict[str, Dict[str, float]]:
    """Look up demographic prior for a US agent, with party shift applied."""
    key = _agent_demo_group(agent)
    base_prior = DEFAULT_DEMO_PRIORS.get(
        key, DEFAULT_DEMO_PRIORS[("metro", "middle", "mid", "White")]
    )

    # Apply party shift
    party = getattr(agent, "political_leaning", "Independent")
    if party not in PARTY_SHIFT:
        party = "Independent"
    shift = PARTY_SHIFT[party]

    pos_neg = {
        "PAGO": ("better", "worse"), "DUR": ("good", "bad"),
        "PEXP": ("better", "worse"), "BUS12": ("good", "bad"), "BUS5": ("good", "bad"),
    }
    neutral = {
        "PAGO": "same", "DUR": "uncertain",
        "PEXP": "same", "BUS12": "uncertain", "BUS5": "uncertain",
    }

    adjusted = {}
    for q in QUESTIONS:
        pos_label, neg_label = pos_neg[q]
        neu_label = neutral[q]
        dp, dn = shift[q]

        p_pos = base_prior[q][pos_label] + dp
        p_neg = base_prior[q][neg_label] + dn
        p_pos = max(0.02, min(0.96, p_pos))
        p_neg = max(0.02, min(0.96, p_neg))
        p_neu = max(0.02, 1.0 - p_pos - p_neg)
        total = p_pos + p_neg + p_neu
        adjusted[q] = {
            pos_label: p_pos / total,
            neg_label: p_neg / total,
            neu_label: p_neu / total,
        }

    return adjusted


def get_demo_group_key(agent) -> DemoGroupKey:
    return _agent_demo_group(agent)
