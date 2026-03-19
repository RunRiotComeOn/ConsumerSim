"""
ConsumerSim US Version — Main Entry Point

Simulates University of Michigan Consumer Sentiment Index (ICS)
using LLM agents with US-specific demographics (race, party, homeownership).

Usage:
  python main_us.py                        # single-step simulation
  python main_us.py --agents 2000          # override agent count
  python main_us.py --no-llm               # skip LLM (synthetic responses)
"""
import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ConsumerSim-US")


def load_config(path: str = "config/config_us.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(__file__), path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_step(config: dict, no_llm: bool = False) -> None:
    """Run a complete single-step US simulation."""
    from src.simulation.engine_us import USSimulationEngine
    from src.evaluation.micro_us import run_micro_validation_us
    from src.evaluation.structural_us import run_structural_validation_us
    from src.visualization.plots import (
        plot_response_distribution,
        plot_demographic_breakdown,
    )
    from src.visualization.plots_us import (
        plot_us_agent_profiles,
        plot_us_validation_summary,
    )

    figures_dir = config.get("output", {}).get("figures_dir", "results_us/figures")
    results_dir = config.get("output", {}).get("results_dir", "results_us")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    engine = USSimulationEngine(config)

    # Calibrate priors to latest real ICS
    if not no_llm:
        engine.calibrate_to_real_ics()

    # Build agents
    engine.build_agents()

    # Plot agent profiles
    plot_us_agent_profiles(
        engine.agents,
        os.path.join(figures_dir, "us_agent_profiles.png")
    )

    # Fetch real-time macro data (FRED + Google Trends)
    if not no_llm:
        logger.info("\n--- Fetching Real-Time Economic Data ---")
        engine.fetch_macro_context()

    if no_llm:
        logger.info("--no-llm flag set: generating synthetic responses for testing...")
        _inject_synthetic_responses(engine.agents, config["simulation"]["random_seed"])
    else:
        # LLM inference on core agents (with real macro context)
        engine.run_llm_inference()

    # Bayesian update + predict normal agents
    engine.update_and_predict()

    # Compute indices
    indices = engine.compute_step_indices()
    logger.info("\n" + "=" * 60)
    logger.info("US CONSUMER SENTIMENT SIMULATION RESULTS:")
    logger.info(f"  ICS (Index of Consumer Sentiment):     {indices['ICS']:.1f}")
    logger.info(f"  ICC (Index of Current Conditions):     {indices['ICC']:.1f}")
    logger.info(f"  ICE (Index of Consumer Expectations):  {indices['ICE']:.1f}")
    logger.info(f"  Component indices:")
    for k in ["PAGO_R", "DUR_R", "PEXP_R", "BUS12_R", "BUS5_R"]:
        logger.info(f"    {k}: {indices[k]:.1f}")
    logger.info("=" * 60)

    # Export results
    results_path = engine.export_results()
    logger.info(f"Results CSV: {results_path}")

    # Visualizations
    plot_response_distribution(
        engine.agents,
        os.path.join(figures_dir, "us_response_distribution.png")
    )

    breakdowns = engine.compute_breakdowns()
    plot_demographic_breakdown(
        breakdowns,
        os.path.join(figures_dir, "us_demographic_breakdown.png")
    )

    # Log breakdowns
    logger.info("\n--- Demographic Breakdowns ---")
    for name, df in breakdowns.items():
        logger.info(f"\n{name}:")
        logger.info(f"\n{df.to_string()}")

    # Validation
    logger.info("\n--- Running US Validation ---")
    micro_df = run_micro_validation_us(engine.agents)
    structural_df = run_structural_validation_us(engine.agents)

    micro_path = os.path.join(results_dir, "us_micro_validation.csv")
    structural_path = os.path.join(results_dir, "us_structural_validation.csv")
    micro_df.to_csv(micro_path)
    structural_df.to_csv(structural_path)

    plot_us_validation_summary(
        micro_df, structural_df,
        os.path.join(figures_dir, "us_validation_summary.png")
    )

    logger.info("\nUS Simulation complete! Results in:")
    logger.info(f"  {results_dir}/")
    logger.info(f"  {figures_dir}/")


def _inject_synthetic_responses(agents, seed: int) -> None:
    """Inject synthetic survey responses for testing without LLM API."""
    import random
    from src.agents.base import SurveyResponse

    rng = random.Random(seed)
    for agent in agents:
        # US-specific optimism factors
        income_factor = (agent.income_percentile / 100) * 0.3
        happiness_factor = (agent.happiness / 3) * 0.25
        confidence_factor = (agent.future_confidence / 5) * 0.25

        # Party effect on macro outlook
        party_boost = 0
        if agent.political_leaning == "Republican":
            party_boost = 0.05   # Slightly more optimistic baseline
        elif agent.political_leaning == "Democrat":
            party_boost = -0.05

        # Race effect
        race_boost = 0
        if agent.race == "White":
            race_boost = 0.05
        elif agent.race == "Black":
            race_boost = -0.08
        elif agent.race == "Asian":
            race_boost = 0.03

        optimism = income_factor + happiness_factor + confidence_factor + race_boost

        def pick_bsw(opt):
            r = rng.random()
            if r < opt * 0.6:
                return "better"
            elif r < opt * 0.6 + 0.4:
                return "same"
            else:
                return "worse"

        def pick_gub(opt, party_adj=0):
            r = rng.random()
            adj_opt = opt + party_adj
            if r < adj_opt * 0.5:
                return "good"
            elif r < adj_opt * 0.5 + 0.4:
                return "uncertain"
            else:
                return "bad"

        agent.response = SurveyResponse(
            PAGO=pick_bsw(optimism),
            DUR=pick_gub(optimism),
            PEXP=pick_bsw(optimism),
            BUS12=pick_gub(optimism, party_boost),
            BUS5=pick_gub(optimism, party_boost),
        )


def run_monthly(config: dict, year: int, month: int, n_agents: int = 2000) -> None:
    """Run simulation for a specific month."""
    from src.simulation.monthly_runner import MonthlyRunner
    runner = MonthlyRunner(config)
    result = runner.run_single_month(year, month, n_agents=n_agents)
    logger.info(f"\nResult: {result}")


def run_backtest(config: dict, start: str = "2024-06",
                  end: str = "2026-01", n_agents: int = 2000) -> None:
    """Run backtest across multiple months."""
    from src.simulation.monthly_runner import MonthlyRunner
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    runner = MonthlyRunner(config)
    df = runner.run_backtest(sy, sm, ey, em, n_agents=n_agents)
    logger.info(f"\nBacktest results:\n{df.to_string()}")


def run_forecast(config: dict, n_agents: int = 2000) -> None:
    """Forecast current month's ICS."""
    from src.simulation.monthly_runner import MonthlyRunner
    runner = MonthlyRunner(config)
    result = runner.forecast_next_month(n_agents=n_agents)
    logger.info(f"\nForecast: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="ConsumerSim US - LLM Agent Michigan Consumer Sentiment Simulation"
    )
    parser.add_argument("--agents", type=int, default=None,
                        help="Override total agent count")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM (use synthetic responses for testing)")
    parser.add_argument("--config", default="config/config_us.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", choices=["single", "monthly", "backtest", "forecast"],
                        default="single",
                        help="Run mode: single (default), monthly (specific month), "
                             "backtest (multi-month), forecast (predict current month)")
    parser.add_argument("--month", type=str, default=None,
                        help="Month for monthly mode (e.g., 2025-06)")
    parser.add_argument("--start", type=str, default="2024-06",
                        help="Start month for backtest (e.g., 2024-06)")
    parser.add_argument("--end", type=str, default="2026-01",
                        help="End month for backtest (e.g., 2026-01)")
    args = parser.parse_args()

    logger.info("ConsumerSim US starting...")
    logger.info("Target: University of Michigan Consumer Sentiment Index (ICS)")
    config = load_config(args.config)

    n_agents = args.agents or config["simulation"]["total_agents"]
    if args.agents:
        config["simulation"]["total_agents"] = args.agents
        logger.info(f"Agent count overridden: {args.agents}")

    if args.mode == "single":
        run_single_step(config, no_llm=args.no_llm)
    elif args.mode == "monthly":
        if not args.month:
            logger.error("--month required for monthly mode (e.g., --month 2025-06)")
            return
        y, m = map(int, args.month.split("-"))
        run_monthly(config, y, m, n_agents=n_agents)
    elif args.mode == "backtest":
        run_backtest(config, start=args.start, end=args.end, n_agents=n_agents)
    elif args.mode == "forecast":
        run_forecast(config, n_agents=n_agents)


if __name__ == "__main__":
    main()
