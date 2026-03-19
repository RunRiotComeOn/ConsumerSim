"""
US Simulation Engine.

Orchestrates the full US ConsumerSim pipeline:
  1. Build synthetic US population
  2. Create agents (core + normal split)
  3. Run LLM inference on core agents (gpt-4o-mini)
  4. Bayesian update with 72 groups + party shift
  5. Sample from posterior for normal agents
  6. Aggregate Michigan-style indices (ICS/ICC/ICE)
"""
import logging
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.agents.base_us import USConsumerAgent
from src.behavior_model.predictor_us import USBayesianPredictor
from src.llm.inference_us import USLLMInferenceEngine
from src.population.builder_us import USPopulationBuilder
from src.simulation.index import compute_indices, compute_indices_by_group

logger = logging.getLogger(__name__)


class USSimulationEngine:
    """Main US simulation controller."""

    def __init__(self, config: dict):
        self.config = config
        self.rng = random.Random(config["simulation"]["random_seed"])
        np.random.seed(config["simulation"]["random_seed"])

        self.population_builder = USPopulationBuilder(config)
        self.llm_engine = USLLMInferenceEngine(config)
        self.macro_context = ""  # Will be filled with real-time data
        self.prediction_target_month = ""
        self.survey_cutoff_date = ""
        self.prior_ics = None  # Last month's real ICS for regime anchoring

        self.agents: List[USConsumerAgent] = []
        self.core_agents: List[USConsumerAgent] = []
        self.normal_agents: List[USConsumerAgent] = []
        self.results: List[Dict] = []

        os.makedirs(config.get("output", {}).get("results_dir", "results_us"), exist_ok=True)
        os.makedirs(config.get("output", {}).get("figures_dir", "results_us/figures"), exist_ok=True)

    def calibrate_to_real_ics(self, target_ics: float = None) -> None:
        """Calibrate priors to the latest real Michigan ICS.

        If target_ics is provided, uses that value directly.
        Otherwise fetches the latest UMCSENT from FRED.
        """
        from src.behavior_model.demo_guidance_us import calibrate_priors

        if target_ics is not None:
            self.prior_ics = target_ics
            calibrate_priors(target_ics)
            return

        try:
            from src.data.fred_collector import FREDCollector
            api_key = self.config["data"].get("fred_api_key", "")
            fred = FREDCollector(api_key)
            data = fred.fetch_series("UMCSENT")
            if len(data) > 0:
                latest_ics = float(data.dropna().iloc[-1])
                self.prior_ics = latest_ics
                logger.info(f"Latest real UMCSENT: {latest_ics:.1f}")
                calibrate_priors(latest_ics)
            else:
                logger.warning("No UMCSENT data available, using default priors")
        except Exception as e:
            logger.warning(f"Failed to fetch UMCSENT for calibration: {e}. Using default priors.")

    def build_agents(self) -> List[USConsumerAgent]:
        n_total = self.config["simulation"]["total_agents"]
        core_ratio = self.config["simulation"]["core_agent_ratio"]
        n_core = max(10, int(n_total * core_ratio))

        logger.info(f"Building {n_total} US agents ({n_core} core, {n_total - n_core} normal)...")

        population_df = self.population_builder.build(n_total)

        core_indices = set(
            self.rng.sample(range(len(population_df)), n_core)
        )

        self.agents = []
        for i, row in population_df.iterrows():
            is_core = i in core_indices
            agent = USConsumerAgent.from_row(row.to_dict(), agent_id=int(i), is_core=is_core)
            self.agents.append(agent)

        self.core_agents = [a for a in self.agents if a.is_core]
        self.normal_agents = [a for a in self.agents if not a.is_core]

        logger.info(
            f"US agents ready: {len(self.core_agents)} core, {len(self.normal_agents)} normal."
        )

        # Log demographic distribution
        races = pd.Series([a.race for a in self.agents]).value_counts(normalize=True)
        parties = pd.Series([a.political_leaning for a in self.agents]).value_counts(normalize=True)
        logger.info(f"  Race distribution: {dict(races.round(3))}")
        logger.info(f"  Party distribution: {dict(parties.round(3))}")

        return self.agents

    def fetch_macro_context(self) -> str:
        """Fetch real-time macro data and build context for LLM prompts."""
        try:
            from src.data.macro_context import MacroContextBuilder
            builder = MacroContextBuilder(self.config)
            self.macro_context = builder.build_context(
                include_fred=True, include_trends=True
            )
            logger.info(f"Macro context built ({len(self.macro_context)} chars)")
            logger.info(self.macro_context[:500] + "..." if len(self.macro_context) > 500 else self.macro_context)
        except Exception as e:
            logger.warning(f"Failed to fetch macro context: {e}. Proceeding without it.")
            self.macro_context = ""

        if not self.prediction_target_month:
            now = datetime.now()
            self.prediction_target_month = f"{now.year}-{now.month:02d}"
        if not self.survey_cutoff_date:
            sim_cfg = self.config.get("simulation", {})
            day = sim_cfg.get("survey_window_cutoff_day", sim_cfg.get("survey_second_week_start_day", 25))
            day = max(1, min(int(day), 28))
            now = datetime.now()
            self.survey_cutoff_date = f"{now.year}-{now.month:02d}-{day:02d}"
        return self.macro_context

    def run_llm_inference(self) -> None:
        logger.info("Running LLM inference on US core agents...")
        self.llm_engine.run_batch(
            self.core_agents,
            macro_context=self.macro_context,
            prediction_target_month=self.prediction_target_month,
            survey_cutoff_date=self.survey_cutoff_date,
            prior_ics=self.prior_ics,
        )
        valid = sum(1 for a in self.core_agents if a.response.is_valid())
        logger.info(f"LLM inference complete. Valid responses: {valid}/{len(self.core_agents)}")
        if valid == 0:
            logger.error("ALL core agent LLM calls failed! Results will be prior-only (no LLM adjustment).")
        elif valid < len(self.core_agents) * 0.5:
            logger.warning(f"Only {valid}/{len(self.core_agents)} valid LLM responses — results may be unreliable.")

    def update_and_predict(self) -> None:
        post_cfg = self.config.get("posterior", {})
        predictor = USBayesianPredictor(
            seed=self.config["simulation"]["random_seed"],
            prior_strength=post_cfg.get("prior_strength", 20),
        )
        predictor.update_posteriors(self.core_agents)
        predictor.predict_batch(self.normal_agents)
        logger.info("Normal agent responses predicted.")

    def compute_step_indices(self) -> Dict:
        indices = compute_indices(self.agents)
        logger.info(
            f"ICS={indices['ICS']:.1f} | "
            f"ICC={indices['ICC']:.1f} | ICE={indices['ICE']:.1f}"
        )
        return indices

    def run_once(self) -> Dict:
        self.calibrate_to_real_ics()
        self.build_agents()
        self.fetch_macro_context()
        self.run_llm_inference()
        self.update_and_predict()
        indices = self.compute_step_indices()
        self.results.append(indices)
        return indices

    def compute_breakdowns(self) -> Dict[str, pd.DataFrame]:
        """Compute ICS broken down by US demographic groups."""
        return {
            "by_region": compute_indices_by_group(self.agents, "region"),
            "by_metro": compute_indices_by_group(self.agents, "metro_status"),
            "by_age": compute_indices_by_group(self.agents, "age_group"),
            "by_income": compute_indices_by_group(self.agents, "income_tercile"),
            "by_race": compute_indices_by_group(self.agents, "race"),
            "by_party": compute_indices_by_group(self.agents, "political_leaning"),
            "by_education": compute_indices_by_group(self.agents, "education"),
        }

    def export_results(self, output_path: str | None = None) -> str:
        if output_path is None:
            output_path = os.path.join(
                self.config.get("output", {}).get("results_dir", "results_us"),
                "us_simulation_results.csv"
            )

        rows = []
        for agent in self.agents:
            row = agent.to_feature_vector()
            row["agent_id"] = agent.agent_id
            row["is_core"] = agent.is_core
            row["state"] = agent.state
            row["race"] = agent.race
            row["political_leaning"] = agent.political_leaning
            row.update(agent.response.to_dict())
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")
        return output_path
