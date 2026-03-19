"""
US Bayesian Posterior Predictor.

Uses 72 demographic groups (metro x age x income x race) + party shift.
Same Dirichlet-Multinomial conjugate update as the Chinese version.
"""
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.agents.base import SurveyResponse
from src.agents.base_us import USConsumerAgent
import src.behavior_model.demo_guidance_us as demo_guidance_us
from src.behavior_model.demo_guidance_us import (
    QUESTIONS,
    DemoGroupKey,
    get_demo_group_key,
    get_demo_prior,
)

logger = logging.getLogger(__name__)

CATEGORIES = {
    "PAGO":  ["better", "same", "worse"],
    "DUR":   ["good", "uncertain", "bad"],
    "PEXP":  ["better", "same", "worse"],
    "BUS12": ["good", "uncertain", "bad"],
    "BUS5":  ["good", "uncertain", "bad"],
}


class USBayesianPredictor:
    """Bayesian predictor for US version with 72 groups + party shift."""

    def __init__(self, seed: int = 42, prior_strength: int = 20):
        self.rng = np.random.default_rng(seed)
        self.prior_strength = prior_strength
        self.posteriors: Dict[DemoGroupKey, Dict[str, Dict[str, float]]] = {}

    def update_posteriors(self, core_agents: List[USConsumerAgent]) -> None:
        """Bayesian update using core agents' LLM responses."""
        PRIOR_STRENGTH = self.prior_strength

        # Filter out agents with invalid (failed LLM) responses
        valid_agents = [a for a in core_agents if a.response.is_valid()]
        n_invalid = len(core_agents) - len(valid_agents)
        if n_invalid > 0:
            logger.warning(f"  Skipping {n_invalid}/{len(core_agents)} core agents with invalid LLM responses")

        groups: Dict[DemoGroupKey, List[USConsumerAgent]] = defaultdict(list)
        for agent in valid_agents:
            key = get_demo_group_key(agent)
            groups[key].append(agent)

        logger.info(f"Updating posteriors from {len(valid_agents)} valid core agents "
                     f"(of {len(core_agents)} total) across {len(groups)} demographic groups (72 possible)...")

        for key in demo_guidance_us.DEFAULT_DEMO_PRIORS:
            prior = demo_guidance_us.DEFAULT_DEMO_PRIORS[key]
            group_agents = groups.get(key, [])
            n_obs = len(group_agents)

            group_posterior: Dict[str, Dict[str, float]] = {}

            for q in QUESTIONS:
                cats = CATEGORIES[q]
                prior_dist = prior[q]

                alpha_prior = {c: prior_dist.get(c, 1 / len(cats)) * PRIOR_STRENGTH
                               for c in cats}

                obs_counts = {c: 0 for c in cats}
                for agent in group_agents:
                    response_val = getattr(agent.response, q)
                    if response_val in obs_counts:
                        obs_counts[response_val] += 1

                alpha_post = {c: alpha_prior[c] + obs_counts[c] for c in cats}
                total = sum(alpha_post.values())
                posterior_dist = {c: alpha_post[c] / total for c in cats}

                group_posterior[q] = posterior_dist

            self.posteriors[key] = group_posterior

            if n_obs > 0:
                example_q = "PAGO"
                cats = CATEGORIES[example_q]
                prior_str = " ".join(f"{c}={prior[example_q].get(c,0):.0%}" for c in cats)
                post_str = " ".join(f"{c}={group_posterior[example_q][c]:.0%}" for c in cats)
                logger.info(f"  {key}: n={n_obs}  "
                            f"PAGO prior=[{prior_str}] -> post=[{post_str}]")

        n_empty = sum(1 for k in demo_guidance_us.DEFAULT_DEMO_PRIORS if k not in groups)
        if n_empty > 0:
            logger.info(f"  {n_empty} groups had no core agents -> prior unchanged.")

    def predict_batch(self, agents: List[USConsumerAgent]) -> List[USConsumerAgent]:
        """Sample survey responses for normal agents from posterior + party shift."""
        if not agents:
            return agents
        if not self.posteriors:
            raise RuntimeError("Must call update_posteriors() before predict_batch().")

        n = len(agents)
        logger.info(f"Sampling responses for {n} normal agents...")

        fallback_key = ("metro", "middle", "mid", "White")

        predictions: Dict[str, list] = {q: [] for q in QUESTIONS}

        for agent in agents:
            key = get_demo_group_key(agent)
            posterior = self.posteriors.get(key, self.posteriors.get(fallback_key, {}))

            # Apply party shift to posterior for this agent
            adjusted = self._apply_party_shift(posterior, agent)

            for q in QUESTIONS:
                cats = CATEGORIES[q]
                dist = adjusted.get(q, {c: 1 / len(cats) for c in cats})
                probs = np.array([dist[c] for c in cats])
                probs = probs / probs.sum()
                choice = self.rng.choice(cats, p=probs)
                predictions[q].append(choice)

        for i, agent in enumerate(agents):
            agent.response = SurveyResponse(
                PAGO=predictions["PAGO"][i],
                DUR=predictions["DUR"][i],
                PEXP=predictions["PEXP"][i],
                BUS12=predictions["BUS12"][i],
                BUS5=predictions["BUS5"][i],
            )

        self._log_distribution_summary(agents)
        logger.info("Normal agent prediction complete.")
        return agents

    def _apply_party_shift(self, posterior, agent):
        """Apply party shift to posterior distributions."""
        from src.behavior_model.demo_guidance_us import PARTY_SHIFT

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
            dist = posterior.get(q, {pos_label: 0.33, neg_label: 0.33, neu_label: 0.34})

            p_pos = dist.get(pos_label, 0.33) + dp
            p_neg = dist.get(neg_label, 0.33) + dn
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

    def _log_distribution_summary(self, agents: List[USConsumerAgent]) -> None:
        n = len(agents)
        for q in QUESTIONS:
            responses = [getattr(a.response, q) for a in agents]
            counts: Dict[str, int] = {}
            for r in responses:
                counts[r] = counts.get(r, 0) + 1
            dist_str = " | ".join(
                f"{k}: {v / n:.1%}" for k, v in sorted(counts.items())
            )
            logger.info(f"  {q}: {dist_str}")
