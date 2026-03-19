"""
Diagnose LLM response quality for a specific month.
Run: python diagnose_llm.py --month 2025-03 --agents 100

This prints:
1. The macro context sent to LLM
2. Core agent LLM responses (distribution)
3. Core-only ICS vs Prior ICS vs Real ICS
4. After Bayesian update ICS
"""
import argparse
import logging
import os
import sys
import yaml
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, required=True, help="e.g. 2025-03")
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--config", default="config/config_us.yaml")
    args = parser.parse_args()

    year, month = map(int, args.month.split("-"))

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["simulation"]["total_agents"] = args.agents

    from src.simulation.monthly_runner import MonthlyRunner
    from src.simulation.engine_us import USSimulationEngine
    from src.behavior_model.demo_guidance_us import calibrate_priors
    from src.simulation.index import compute_indices

    runner = MonthlyRunner(config)

    # 1. Get prior month ICS
    prior_ics = runner.get_prior_month_ics(year, month)
    logger.info(f"Prior month ICS: {prior_ics}")

    # 2. Build context
    context = runner.build_month_context(year, month)
    print("\n" + "=" * 70)
    print("MACRO CONTEXT SENT TO LLM:")
    print("=" * 70)
    print(context)
    print("=" * 70 + "\n")

    # 3. Calibrate and build
    engine = USSimulationEngine(config)
    engine.prediction_target_month = f"{year}-{month:02d}"
    engine.survey_cutoff_date = runner._survey_cutoff(year, month).strftime("%Y-%m-%d")
    if prior_ics:
        engine.calibrate_to_real_ics(target_ics=prior_ics)
    engine.build_agents()
    engine.macro_context = context

    # 4. LLM inference (prior_ics flows through engine)
    engine.run_llm_inference()

    # 5. Analyze core responses
    print("\n" + "=" * 70)
    print("CORE AGENT LLM RESPONSES:")
    print("=" * 70)
    for q in ["PAGO", "DUR", "PEXP", "BUS12", "BUS5"]:
        responses = [getattr(a.response, q) for a in engine.core_agents]
        counts = Counter(responses)
        n = len(responses)
        dist_str = " | ".join(
            f"{k}: {v}/{n} ({v / n:.0%})" for k, v in sorted(counts.items())
        )
        print(f"  {q}: {dist_str}")

    # Compute Xi for each question from core agents
    print("\n  Component Xi values (core only):")
    for q in ["PAGO", "DUR", "PEXP", "BUS12", "BUS5"]:
        responses = [getattr(a.response, q) for a in engine.core_agents]
        n = len(responses)
        if q in ("PAGO", "PEXP"):
            pos = responses.count("better") / n
            neg = responses.count("worse") / n
        else:
            pos = responses.count("good") / n
            neg = responses.count("bad") / n
        xi = (pos - neg) * 100 + 100
        print(f"    {q}: pos={pos:.0%} neg={neg:.0%} → Xi={xi:.1f}")

    core_indices = compute_indices(engine.core_agents)
    print(f"\n  Core-only ICS: {core_indices['ICS']:.1f}")
    print(f"  Prior ICS:     {prior_ics}")

    # 6. Bayesian update
    engine.update_and_predict()
    all_indices = compute_indices(engine.agents)
    print(f"  Final ICS:     {all_indices['ICS']:.1f}")

    # Fetch real ICS for target month
    try:
        from src.data.fred_collector import FREDCollector
        fred = FREDCollector(config["data"].get("fred_api_key", ""))
        real_data = fred.fetch_series("UMCSENT", start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-28")
        if len(real_data) > 0:
            real_ics = float(real_data.iloc[-1])
            print(f"  Real ICS:      {real_ics:.1f}")
        else:
            print(f"  Real ICS:      not available")
    except Exception:
        print(f"  Real ICS:      fetch failed")

    # 7. Sample a few core agent prompts to see what LLM actually saw
    print("\n" + "=" * 70)
    print("SAMPLE CORE AGENT PROFILES (first 3):")
    print("=" * 70)
    for i, agent in enumerate(engine.core_agents[:3]):
        print(f"\n--- Agent {i} ---")
        print(agent.to_profile_text())
        r = agent.response
        print(f"  → LLM response: PAGO={r.PAGO} DUR={r.DUR} PEXP={r.PEXP} BUS12={r.BUS12} BUS5={r.BUS5}")


if __name__ == "__main__":
    main()
