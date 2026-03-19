"""
US Population Builder: synthesizes agent profiles from ACS/CPS/GSS distributions.

Generates a realistic synthetic US population with properly aligned JOINT
distributions across demographics, not just independent marginals.

Key joint distributions modeled:
  - Race × Party (Black ~85% Dem, Hispanic ~60% Dem, White ~42% Rep)
  - Race × Education (Asian ~60% BA+, Hispanic ~20% BA+)
  - State × Race (state-level racial composition from ACS)
  - Age × Income (peak at 45-54, decline after 65)
  - Education × Party (graduates lean Democrat)
  - Education × Income (graduate premium)
  - Race × Homeownership (Black/Hispanic ~20pt gap)
"""
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- ACS/CPS-based marginal distributions ---

# State populations (2023 ACS estimates)
STATE_WEIGHTS = {
    "California": 0.118, "Texas": 0.090, "Florida": 0.069, "New York": 0.059,
    "Pennsylvania": 0.039, "Illinois": 0.038, "Ohio": 0.035, "Georgia": 0.033,
    "North Carolina": 0.032, "Michigan": 0.030, "New Jersey": 0.028,
    "Virginia": 0.026, "Washington": 0.024, "Arizona": 0.023,
    "Tennessee": 0.022, "Massachusetts": 0.021, "Indiana": 0.020,
    "Maryland": 0.019, "Missouri": 0.018, "Wisconsin": 0.018,
    "Colorado": 0.018, "Minnesota": 0.017, "South Carolina": 0.016,
    "Alabama": 0.015, "Louisiana": 0.014, "Kentucky": 0.014,
    "Oregon": 0.013, "Oklahoma": 0.012, "Connecticut": 0.011,
    "Iowa": 0.010, "Mississippi": 0.009, "Arkansas": 0.009,
    "Nevada": 0.010, "Utah": 0.010, "Kansas": 0.009,
    "New Mexico": 0.006, "Nebraska": 0.006, "Idaho": 0.006,
    "West Virginia": 0.005, "Hawaii": 0.004, "New Hampshire": 0.004,
    "Maine": 0.004, "Montana": 0.003, "Rhode Island": 0.003,
    "Delaware": 0.003, "South Dakota": 0.003, "North Dakota": 0.002,
    "Alaska": 0.002, "Vermont": 0.002, "Wyoming": 0.002,
    "District of Columbia": 0.002,
}

# --- JOINT DISTRIBUTION: Race × Party (CES/CCES 2022, n=54,828) ---
# P(Party | Race) — from Cooperative Election Study 2022
PARTY_BY_RACE = {
    "White":    {"Democrat": 0.364, "Independent": 0.300, "Republican": 0.336},
    "Black":    {"Democrat": 0.709, "Independent": 0.225, "Republican": 0.065},
    "Hispanic": {"Democrat": 0.457, "Independent": 0.306, "Republican": 0.237},
    "Asian":    {"Democrat": 0.512, "Independent": 0.342, "Republican": 0.146},
    "Other":    {"Democrat": 0.410, "Independent": 0.369, "Republican": 0.221},
}

# --- JOINT DISTRIBUTION: Race × Education (ACS 2023, 25+) ---
# P(Education | Race)
EDU_BY_RACE = {
    "White":    {"less_than_hs": 0.08, "high_school": 0.26, "some_college": 0.20,
                 "bachelors": 0.25, "graduate": 0.21},
    "Black":    {"less_than_hs": 0.12, "high_school": 0.30, "some_college": 0.25,
                 "bachelors": 0.20, "graduate": 0.13},
    "Hispanic": {"less_than_hs": 0.25, "high_school": 0.28, "some_college": 0.22,
                 "bachelors": 0.16, "graduate": 0.09},
    "Asian":    {"less_than_hs": 0.10, "high_school": 0.14, "some_college": 0.10,
                 "bachelors": 0.33, "graduate": 0.33},
    "Other":    {"less_than_hs": 0.13, "high_school": 0.28, "some_college": 0.24,
                 "bachelors": 0.21, "graduate": 0.14},
}

# --- JOINT DISTRIBUTION: State × Race (ACS 2023, top states) ---
# P(Race | State) — states not listed use national average
STATE_RACE_DIST = {
    "California":   {"White": 0.35, "Hispanic": 0.40, "Black": 0.06, "Asian": 0.16, "Other": 0.03},
    "Texas":        {"White": 0.40, "Hispanic": 0.40, "Black": 0.12, "Asian": 0.05, "Other": 0.03},
    "Florida":      {"White": 0.51, "Hispanic": 0.27, "Black": 0.15, "Asian": 0.03, "Other": 0.04},
    "New York":     {"White": 0.52, "Hispanic": 0.20, "Black": 0.14, "Asian": 0.09, "Other": 0.05},
    "Georgia":      {"White": 0.50, "Hispanic": 0.10, "Black": 0.32, "Asian": 0.04, "Other": 0.04},
    "North Carolina": {"White": 0.60, "Hispanic": 0.10, "Black": 0.21, "Asian": 0.03, "Other": 0.06},
    "Illinois":     {"White": 0.55, "Hispanic": 0.18, "Black": 0.14, "Asian": 0.06, "Other": 0.07},
    "Pennsylvania": {"White": 0.73, "Hispanic": 0.08, "Black": 0.11, "Asian": 0.04, "Other": 0.04},
    "Ohio":         {"White": 0.77, "Hispanic": 0.04, "Black": 0.13, "Asian": 0.02, "Other": 0.04},
    "Michigan":     {"White": 0.73, "Hispanic": 0.06, "Black": 0.14, "Asian": 0.03, "Other": 0.04},
    "New Jersey":   {"White": 0.52, "Hispanic": 0.22, "Black": 0.13, "Asian": 0.10, "Other": 0.03},
    "Virginia":     {"White": 0.58, "Hispanic": 0.10, "Black": 0.19, "Asian": 0.07, "Other": 0.06},
    "Maryland":     {"White": 0.47, "Hispanic": 0.11, "Black": 0.30, "Asian": 0.07, "Other": 0.05},
    "Mississippi":  {"White": 0.55, "Hispanic": 0.03, "Black": 0.37, "Asian": 0.01, "Other": 0.04},
    "Alabama":      {"White": 0.63, "Hispanic": 0.05, "Black": 0.26, "Asian": 0.02, "Other": 0.04},
    "South Carolina": {"White": 0.62, "Hispanic": 0.06, "Black": 0.26, "Asian": 0.02, "Other": 0.04},
    "Louisiana":    {"White": 0.56, "Hispanic": 0.07, "Black": 0.32, "Asian": 0.02, "Other": 0.03},
    "Tennessee":    {"White": 0.72, "Hispanic": 0.06, "Black": 0.16, "Asian": 0.02, "Other": 0.04},
    "Arizona":      {"White": 0.53, "Hispanic": 0.32, "Black": 0.05, "Asian": 0.04, "Other": 0.06},
    "Nevada":       {"White": 0.45, "Hispanic": 0.30, "Black": 0.10, "Asian": 0.10, "Other": 0.05},
    "New Mexico":   {"White": 0.35, "Hispanic": 0.49, "Black": 0.02, "Asian": 0.02, "Other": 0.12},
    "Hawaii":       {"White": 0.22, "Hispanic": 0.11, "Black": 0.02, "Asian": 0.37, "Other": 0.28},
    "District of Columbia": {"White": 0.37, "Hispanic": 0.11, "Black": 0.44, "Asian": 0.04, "Other": 0.04},
    "Vermont":      {"White": 0.92, "Hispanic": 0.02, "Black": 0.01, "Asian": 0.02, "Other": 0.03},
    "Maine":        {"White": 0.91, "Hispanic": 0.02, "Black": 0.02, "Asian": 0.01, "Other": 0.04},
    "West Virginia": {"White": 0.91, "Hispanic": 0.02, "Black": 0.04, "Asian": 0.01, "Other": 0.02},
    "Idaho":        {"White": 0.80, "Hispanic": 0.13, "Black": 0.01, "Asian": 0.02, "Other": 0.04},
    "Montana":      {"White": 0.85, "Hispanic": 0.04, "Black": 0.01, "Asian": 0.01, "Other": 0.09},
    "Washington":   {"White": 0.60, "Hispanic": 0.14, "Black": 0.04, "Asian": 0.10, "Other": 0.12},
    "Massachusetts": {"White": 0.67, "Hispanic": 0.13, "Black": 0.08, "Asian": 0.07, "Other": 0.05},
    "Colorado":     {"White": 0.65, "Hispanic": 0.22, "Black": 0.04, "Asian": 0.04, "Other": 0.05},
    "Minnesota":    {"White": 0.77, "Hispanic": 0.06, "Black": 0.07, "Asian": 0.05, "Other": 0.05},
    "Wisconsin":    {"White": 0.78, "Hispanic": 0.08, "Black": 0.06, "Asian": 0.03, "Other": 0.05},
}

# National average race distribution (fallback for states not in STATE_RACE_DIST)
RACE_WEIGHTS = {"White": 0.58, "Hispanic": 0.19, "Black": 0.13, "Asian": 0.06, "Other": 0.04}

# --- Education × Party adjustment (CES 2022, n=54,828) ---
# Additional party shift by education (applied on top of race-based party)
# Graduates lean more Democrat; non-college lean more Republican
EDU_PARTY_SHIFT = {
    "less_than_hs": {"Democrat": -0.075, "Independent": 0.048, "Republican": 0.027},
    "high_school":  {"Democrat": -0.050, "Independent": -0.020, "Republican": 0.070},
    "some_college": {"Democrat": -0.012, "Independent": 0.009, "Republican": 0.003},
    "bachelors":    {"Democrat": 0.038, "Independent": 0.004, "Republican": -0.041},
    "graduate":     {"Democrat": 0.095, "Independent": -0.004, "Republican": -0.091},
}

# Employment (CPS 2024, civilian 16+)
EMPLOY_WEIGHTS = {"employed": 0.60, "unemployed": 0.04, "retired": 0.20, "other": 0.16}

# Health status (BRFSS)
HEALTH_WEIGHTS = {"excellent": 0.18, "good": 0.52, "fair": 0.22, "poor": 0.08}

# Metro/non-metro (ACS)
METRO_WEIGHTS = {"metro": 0.86, "non-metro": 0.14}

# Homeownership rate (ACS ~65%)
HOMEOWNER_RATE = 0.65

# Health insurance coverage (~92%)
INSURED_RATE = 0.92

# Student debt (among 18-45: ~35%)
STUDENT_DEBT_RATE_YOUNG = 0.35
STUDENT_DEBT_RATE_OLD = 0.05

# Income by race (median household, for generating distributions)
INCOME_PARAMS = {
    "White":    {"mean": 75000, "std": 45000},
    "Black":    {"mean": 52000, "std": 35000},
    "Hispanic": {"mean": 57000, "std": 38000},
    "Asian":    {"mean": 105000, "std": 60000},
    "Other":    {"mean": 62000, "std": 40000},
}

# Age × Income multiplier (CPS: earnings peak at 45-54, drop after 65)
AGE_INCOME_MULT = {
    # (min_age, max_age): multiplier on base income
    (18, 24): 0.45,
    (25, 34): 0.80,
    (35, 44): 1.00,
    (45, 54): 1.10,  # peak earning years
    (55, 64): 0.95,
    (65, 85): 0.50,  # retirement income drop
}

# ============================================================
# GSS 2022 fitted parameters (replaces hand-tuned formulas)
# All conditional distributions below are from GSS 2022 microdata
# ============================================================

# Happiness (1-3 scale, 3=very happy) by Race × Income tercile
# GSS variable: HAPPY (reversed so 3=very happy)
GSS_HAPPINESS = {
    ("White", "low"):    {"mean": 1.772, "std": 0.640},
    ("White", "mid"):    {"mean": 1.972, "std": 0.661},
    ("White", "high"):   {"mean": 2.155, "std": 0.626},
    ("Black", "low"):    {"mean": 1.788, "std": 0.707},
    ("Black", "mid"):    {"mean": 1.903, "std": 0.647},
    ("Black", "high"):   {"mean": 2.131, "std": 0.635},
    ("Hispanic", "low"): {"mean": 1.907, "std": 0.721},
    ("Hispanic", "mid"): {"mean": 1.993, "std": 0.694},
    ("Hispanic", "high"):{"mean": 2.164, "std": 0.646},
    ("Asian", "low"):    {"mean": 1.725, "std": 0.616},  # using Other low
    ("Asian", "mid"):    {"mean": 1.867, "std": 0.692},  # using Other mid
    ("Asian", "high"):   {"mean": 2.010, "std": 0.616},  # using Other high
    ("Other", "low"):    {"mean": 1.725, "std": 0.616},
    ("Other", "mid"):    {"mean": 1.867, "std": 0.692},
    ("Other", "high"):   {"mean": 2.010, "std": 0.616},
}

# Financial satisfaction (1-3 scale, 3=satisfied) by Race × Income tercile
# GSS variable: SATFIN (reversed so 3=satisfied)
GSS_SATFIN = {
    ("White", "low"):    {"mean": 1.602, "std": 0.692},
    ("White", "mid"):    {"mean": 1.937, "std": 0.728},
    ("White", "high"):   {"mean": 2.304, "std": 0.674},
    ("Black", "low"):    {"mean": 1.634, "std": 0.714},
    ("Black", "mid"):    {"mean": 1.820, "std": 0.713},
    ("Black", "high"):   {"mean": 2.078, "std": 0.748},
    ("Hispanic", "low"): {"mean": 1.667, "std": 0.635},
    ("Hispanic", "mid"): {"mean": 1.797, "std": 0.655},
    ("Hispanic", "high"):{"mean": 2.100, "std": 0.707},
    ("Asian", "low"):    {"mean": 1.700, "std": 0.709},
    ("Asian", "mid"):    {"mean": 1.953, "std": 0.698},
    ("Asian", "high"):   {"mean": 2.137, "std": 0.636},
    ("Other", "low"):    {"mean": 1.700, "std": 0.709},
    ("Other", "mid"):    {"mean": 1.953, "std": 0.698},
    ("Other", "high"):   {"mean": 2.137, "std": 0.636},
}

# Trust rate by Race
# GSS variable: TRUST (P(can trust most people))
GSS_TRUST_RATE = {
    "White": 0.328, "Black": 0.165, "Hispanic": 0.110,
    "Asian": 0.214, "Other": 0.214,  # Asian uses Other rate
}

# Job satisfaction (1-4 scale, 4=very satisfied) by employment status
# GSS variable: SATJOB (reversed so 4=very satisfied)
GSS_SATJOB = {
    "employed":   {"mean": 3.277, "std": 0.772},
    "unemployed": {"mean": 2.833, "std": 0.960},
    "retired":    {"mean": 3.200, "std": 0.890},  # using "other" as proxy
    "other":      {"mean": 3.197, "std": 0.893},
}

# Work reward belief (1-3 scale, 3=hard work) by Race
# GSS variable: GETAHEAD (reversed so 3=hard work leads to success)
GSS_GETAHEAD = {
    "White":    {"mean": 2.506, "std": 0.696},
    "Black":    {"mean": 2.410, "std": 0.766},
    "Hispanic": {"mean": 2.477, "std": 0.703},
    "Asian":    {"mean": 2.288, "std": 0.697},  # using Other
    "Other":    {"mean": 2.288, "std": 0.697},
}

# Life excitement (1-3 scale, 3=exciting) by Age group
# GSS variable: LIFE (reversed so 3=exciting)
GSS_LIFE = {
    (18, 30): {"mean": 2.301, "std": 0.578},
    (30, 45): {"mean": 2.373, "std": 0.566},
    (45, 55): {"mean": 2.338, "std": 0.577},
    (55, 65): {"mean": 2.400, "std": 0.554},
    (65, 90): {"mean": 2.406, "std": 0.589},
}

# Health distribution by Race × Income tercile
# GSS variable: HEALTH (1=excellent, 2=good, 3=fair, 4=poor)
GSS_HEALTH = {
    ("White", "low"):    {"excellent": 0.101, "good": 0.443, "fair": 0.351, "poor": 0.105},
    ("White", "mid"):    {"excellent": 0.198, "good": 0.556, "fair": 0.193, "poor": 0.054},
    ("White", "high"):   {"excellent": 0.257, "good": 0.568, "fair": 0.157, "poor": 0.018},
    ("Black", "low"):    {"excellent": 0.129, "good": 0.423, "fair": 0.376, "poor": 0.072},
    ("Black", "mid"):    {"excellent": 0.167, "good": 0.543, "fair": 0.266, "poor": 0.025},
    ("Black", "high"):   {"excellent": 0.235, "good": 0.529, "fair": 0.203, "poor": 0.033},
    ("Hispanic", "low"): {"excellent": 0.178, "good": 0.416, "fair": 0.327, "poor": 0.078},
    ("Hispanic", "mid"): {"excellent": 0.169, "good": 0.551, "fair": 0.243, "poor": 0.037},
    ("Hispanic", "high"):{"excellent": 0.214, "good": 0.537, "fair": 0.234, "poor": 0.015},
    ("Asian", "low"):    {"excellent": 0.155, "good": 0.324, "fair": 0.408, "poor": 0.113},
    ("Asian", "mid"):    {"excellent": 0.119, "good": 0.662, "fair": 0.159, "poor": 0.060},
    ("Asian", "high"):   {"excellent": 0.222, "good": 0.567, "fair": 0.207, "poor": 0.005},
    ("Other", "low"):    {"excellent": 0.155, "good": 0.324, "fair": 0.408, "poor": 0.113},
    ("Other", "mid"):    {"excellent": 0.119, "good": 0.662, "fair": 0.159, "poor": 0.060},
    ("Other", "high"):   {"excellent": 0.222, "good": 0.567, "fair": 0.207, "poor": 0.005},
}


def _norm(weights):
    """Normalize probability array to sum to 1."""
    vals = np.array(list(weights.values()))
    return vals / vals.sum()


def _get_age_income_mult(age: int) -> float:
    """Get income multiplier for a given age."""
    for (lo, hi), mult in AGE_INCOME_MULT.items():
        if lo <= age <= hi:
            return mult
    return 1.0


class USPopulationBuilder:
    """Builds a synthetic US consumer population with aligned joint distributions.

    Fetches real demographic data from Census API when available,
    falling back to hardcoded distributions on failure.
    """

    def __init__(self, config: dict):
        self.config = config
        self._census_race_dist = None
        self._census_rates = None
        self._census_income_by_race = None
        self._census_state_income = None
        self._census_year = None

    def _resolve_survey_month(self) -> tuple[int, int]:
        """Return survey month from config, or fall back to local current month."""
        sim_cfg = self.config.get("simulation", {})
        year = sim_cfg.get("survey_year")
        month = sim_cfg.get("survey_month")
        if isinstance(year, int) and isinstance(month, int) and 1 <= month <= 12:
            return year, month
        now = datetime.now()
        return now.year, now.month

    def _resolve_census_year(self) -> int:
        """Pick the latest likely-available ACS year before survey month.

        ACS 1-year data for year Y is typically released around Sep of Y+1.
        So for a survey in month m of year t:
          - if m >= 9, use t-1
          - else use t-2
        """
        year, month = self._resolve_survey_month()
        if month >= 9:
            return year - 1
        return year - 2

    def _fetch_census_data(self) -> None:
        """Fetch real demographic distributions from Census API."""
        if self._census_race_dist is not None:
            return  # already fetched

        self._census_year = self._resolve_census_year()
        cache_dir = self.config["data"].get("cache_dir", "data/cache")
        race_cache = os.path.join(cache_dir, f"census_state_race_{self._census_year}.json")
        rates_cache = os.path.join(cache_dir, f"census_national_rates_{self._census_year}.json")

        # Try loading from cache first
        import json
        if os.path.exists(race_cache) and os.path.exists(rates_cache):
            try:
                with open(race_cache, "r") as f:
                    self._census_race_dist = json.load(f)
                with open(rates_cache, "r") as f:
                    cached = json.load(f)
                    if "rates" in cached:
                        self._census_rates = cached["rates"]
                        self._census_income_by_race = cached.get("income_by_race")
                        self._census_state_income = cached.get("state_income")
                    else:
                        # Old cache format
                        self._census_rates = cached
                logger.info(f"Census data loaded from cache "
                           f"({len(self._census_race_dist)} states)")
                return
            except Exception:
                pass

        # Fetch from Census API
        api_key = self.config["data"].get("census_api_key", "")
        if not api_key:
            logger.warning("No Census API key, using hardcoded distributions")
            return

        try:
            from src.data.census_collector import CensusCollector
            collector = CensusCollector(api_key)

            self._census_race_dist = collector.get_state_race_distributions(year=self._census_year)
            self._census_rates = collector.get_national_rates(year=self._census_year)
            self._census_income_by_race = collector.get_income_params_by_race(year=self._census_year)
            self._census_state_income = collector.get_state_median_income_map(year=self._census_year)

            # Cache results
            os.makedirs(cache_dir, exist_ok=True)
            with open(race_cache, "w") as f:
                json.dump(self._census_race_dist, f, indent=2)
            with open(rates_cache, "w") as f:
                json.dump({
                    "rates": self._census_rates,
                    "income_by_race": self._census_income_by_race,
                    "state_income": self._census_state_income,
                }, f, indent=2)
            logger.info(
                f"Census data fetched and cached for ACS {self._census_year} "
                f"({len(self._census_race_dist)} states)"
            )

        except Exception as e:
            logger.warning(f"Census API fetch failed: {e}. Using hardcoded distributions.")
            self._census_race_dist = None
            self._census_rates = None

    def _get_state_race_dist(self, state: str) -> dict:
        """Get race distribution for a state (Census API or fallback)."""
        if self._census_race_dist and state in self._census_race_dist:
            return self._census_race_dist[state]
        return STATE_RACE_DIST.get(state, RACE_WEIGHTS)

    def _get_homeownership_rate(self) -> float:
        """Get national homeownership rate (Census API or fallback)."""
        if self._census_rates and "homeownership" in self._census_rates:
            return self._census_rates["homeownership"]
        return HOMEOWNER_RATE

    def _get_income_params(self, race: str) -> dict:
        """Get income distribution params for a race (Census API or fallback)."""
        if self._census_income_by_race and race in self._census_income_by_race:
            return self._census_income_by_race[race]
        return INCOME_PARAMS.get(race, INCOME_PARAMS["Other"])

    def build(self, n_agents: int = None) -> pd.DataFrame:
        if n_agents is None:
            n_agents = self.config["simulation"]["total_agents"]

        cache_dir = self.config["data"].get("cache_dir", "data/cache")
        census_year = self._resolve_census_year()
        cache_path = os.path.join(cache_dir, f"us_population_acs{census_year}.parquet")

        if os.path.exists(cache_path):
            logger.info("Loading US population from cache.")
            pop = pd.read_parquet(cache_path)
            if len(pop) == n_agents:
                return pop

        logger.info(f"Building synthetic US population: {n_agents} agents...")
        os.makedirs(cache_dir, exist_ok=True)

        # Fetch Census data before generating
        self._fetch_census_data()

        rng = np.random.default_rng(self.config["simulation"]["random_seed"])
        pop = self._generate(n_agents, rng)

        pop.to_parquet(cache_path, index=False)
        logger.info(f"US population cached to {cache_path}")
        return pop

    def _generate(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        records = []

        # Step 1: Sample state (marginal)
        states = rng.choice(
            list(STATE_WEIGHTS.keys()),
            size=n, p=_norm(STATE_WEIGHTS),
        )

        # Step 2: Sample race CONDITIONAL on state (Census API or fallback)
        races = []
        for state in states:
            race_dist = self._get_state_race_dist(state)
            race_keys = list(race_dist.keys())
            race_probs = _norm(race_dist)
            races.append(rng.choice(race_keys, p=race_probs))
        races = np.array(races)

        # Step 3: Sample education CONDITIONAL on race
        educations = []
        for race in races:
            edu_dist = EDU_BY_RACE.get(race, EDU_BY_RACE["Other"])
            edu_keys = list(edu_dist.keys())
            edu_probs = _norm(edu_dist)
            educations.append(rng.choice(edu_keys, p=edu_probs))
        educations = np.array(educations)

        # Step 4: Sample party CONDITIONAL on race × education
        parties = []
        for race, edu in zip(races, educations):
            base_party = PARTY_BY_RACE.get(race, PARTY_BY_RACE["Other"])
            edu_shift = EDU_PARTY_SHIFT.get(edu, EDU_PARTY_SHIFT["some_college"])
            # Apply education shift on top of race-based party
            adjusted = {}
            for p in ("Democrat", "Independent", "Republican"):
                adjusted[p] = max(0.01, base_party[p] + edu_shift[p])
            party_keys = list(adjusted.keys())
            party_probs = np.array(list(adjusted.values()))
            party_probs = party_probs / party_probs.sum()
            parties.append(rng.choice(party_keys, p=party_probs))
        parties = np.array(parties)

        # Independent marginals
        metros = rng.choice(
            list(METRO_WEIGHTS.keys()),
            size=n, p=list(METRO_WEIGHTS.values()),
        )
        employs = rng.choice(
            list(EMPLOY_WEIGHTS.keys()),
            size=n, p=list(EMPLOY_WEIGHTS.values()),
        )
        genders = rng.choice(["male", "female"], size=n, p=[0.49, 0.51])

        # Age: realistic US distribution (18-85)
        ages = np.clip(rng.normal(45, 17, size=n), 18, 85).astype(int)

        for i in range(n):
            race = races[i]
            age = ages[i]
            edu = educations[i]
            employ = employs[i]

            # Adjust employment by age
            if age >= 65:
                employ = rng.choice(
                    ["retired", "employed", "other"], p=[0.70, 0.20, 0.10]
                )
            elif age < 22 and edu in ("less_than_hs", "high_school"):
                employ = rng.choice(
                    ["employed", "unemployed", "other"], p=[0.55, 0.15, 0.30]
                )

            # Income: race × education × age (joint, Census-calibrated)
            inc_params = self._get_income_params(race)
            edu_mult = {
                "less_than_hs": 0.55, "high_school": 0.75,
                "some_college": 0.90, "bachelors": 1.30, "graduate": 1.65,
            }[edu]
            age_mult = _get_age_income_mult(age)
            if employ == "unemployed":
                edu_mult *= 0.3
            elif employ == "retired":
                edu_mult *= 0.55

            income = max(0, rng.normal(inc_params["mean"] * edu_mult * age_mult,
                                       inc_params["std"] * 0.6))

            # Homeownership: age × race × income (joint)
            ho_prob = self._get_homeownership_rate()
            if age < 30:
                ho_prob -= 0.30
            elif age < 35:
                ho_prob -= 0.15
            elif age > 55:
                ho_prob += 0.10
            if race == "Black":
                ho_prob -= 0.20  # ACS: Black homeownership ~44% vs White ~73%
            elif race == "Hispanic":
                ho_prob -= 0.15  # ACS: Hispanic ~50%
            elif race == "Asian":
                ho_prob -= 0.05  # ACS: Asian ~63%
            if income > 100000:
                ho_prob += 0.15
            elif income > 70000:
                ho_prob += 0.05
            elif income < 30000:
                ho_prob -= 0.20
            ho_prob = np.clip(ho_prob, 0.05, 0.95)
            homeownership = "owner" if rng.random() < ho_prob else "renter"

            # Debt
            has_loan = 1 if homeownership == "owner" and rng.random() < 0.65 else 0
            debt_amount = max(0, rng.normal(250000, 120000)) if has_loan else 0.0

            # Student debt: age × education (joint)
            sd_rate = STUDENT_DEBT_RATE_YOUNG if age < 45 else STUDENT_DEBT_RATE_OLD
            if edu in ("bachelors", "graduate"):
                sd_rate += 0.15
            elif edu == "some_college":
                sd_rate += 0.05
            has_student_debt = 1 if rng.random() < sd_rate else 0

            # Health insurance: income × race (joint)
            ins_prob = INSURED_RATE
            if income < 25000:
                ins_prob -= 0.10
            if race == "Hispanic":
                ins_prob -= 0.08  # ACS: Hispanic uninsured rate ~18%
            has_insurance = 1 if rng.random() < np.clip(ins_prob, 0.5, 0.99) else 0

            # --- Subjective perceptions from GSS 2022 fitted parameters ---
            # Income tercile for GSS lookup
            inc_terc = "low" if income < 35000 else ("high" if income > 80000 else "mid")
            gss_race = race if race in ("White", "Black", "Hispanic", "Asian") else "Other"

            # Happiness (1-3, 3=very happy) — GSS: Race × Income
            hap = GSS_HAPPINESS.get((gss_race, inc_terc), GSS_HAPPINESS[("Other", "mid")])
            happiness = np.clip(rng.normal(hap["mean"], hap["std"]), 1, 3)

            # Financial satisfaction (1-3 → scaled to 1-5) — GSS: Race × Income
            sf = GSS_SATFIN.get((gss_race, inc_terc), GSS_SATFIN[("Other", "mid")])
            # GSS satfin is 1-3, our agent uses 1-5; scale: (gss-1)/2*4+1
            satfin_raw = np.clip(rng.normal(sf["mean"], sf["std"]), 1, 3)
            fin_sat = 1 + (satfin_raw - 1) * 2  # maps 1→1, 2→3, 3→5

            # Life excitement as proxy for life satisfaction (1-3 → 1-5) — GSS: Age
            life_params = GSS_LIFE.get((18, 30), GSS_LIFE[(30, 45)])  # default
            for (lo, hi), params in GSS_LIFE.items():
                if lo <= age < hi:
                    life_params = params
                    break
            life_raw = np.clip(rng.normal(life_params["mean"], life_params["std"]), 1, 3)
            life_sat = 1 + (life_raw - 1) * 2  # maps 1→1, 2→3, 3→5

            # Future confidence: derived from life_sat + age effect (younger more optimistic)
            # No direct GSS variable, but anchored to life satisfaction with age shift
            age_conf_shift = 0.3 if age < 35 else (-0.3 if age > 60 else 0.0)
            unemp_adj = -0.8 if employ == "unemployed" else 0.0
            future_conf = np.clip(life_sat + age_conf_shift + unemp_adj + rng.normal(0, 0.5), 1, 5)

            # Income local rank — derived from actual income percentile
            # Will be computed after all agents are generated (using rank)
            # For now, use a rough estimate
            inc_rank = np.clip(1 + (income / 40000), 1, 5)

            # Social status — GSS CLASS_ proxy: Race × Education × Income
            edu_class = {"less_than_hs": 1.8, "high_school": 2.3,
                         "some_college": 2.6, "bachelors": 3.2, "graduate": 3.7}[edu]
            inc_class_adj = min(income / 100000, 1.0) * 0.8
            social_stat = np.clip(rng.normal(edu_class + inc_class_adj, 0.6), 1, 5)

            # Job satisfaction (1-4, 4=very satisfied) — GSS: Employment status
            js = GSS_SATJOB.get(employ, GSS_SATJOB["other"])
            job_sat = np.clip(rng.normal(js["mean"], js["std"]), 1, 4)

            # Work reward belief (1-3 → 1-4) — GSS: Race
            ga = GSS_GETAHEAD.get(gss_race, GSS_GETAHEAD["Other"])
            getahead_raw = np.clip(rng.normal(ga["mean"], ga["std"]), 1, 3)
            work_belief = 1 + (getahead_raw - 1) * 1.5  # maps 1→1, 2→2.5, 3→4

            # Mobility belief: correlated with work reward belief + race adjustment
            mob_adj = -0.2 if race in ("Black", "Hispanic") else 0.0
            mob_belief = np.clip(work_belief + mob_adj + rng.normal(0, 0.3), 1, 4)

            # Social trust — GSS: Race
            trust_rate = GSS_TRUST_RATE.get(gss_race, 0.214)
            social_trust = "trust" if rng.random() < trust_rate else "cautious"

            # Health — GSS: Race × Income (categorical distribution)
            health_dist = GSS_HEALTH.get((gss_race, inc_terc), GSS_HEALTH[("Other", "mid")])
            health_keys = list(health_dist.keys())
            health_probs = np.array(list(health_dist.values()))
            health_probs = health_probs / health_probs.sum()
            health_status = rng.choice(health_keys, p=health_probs)

            # Asset level: income + age
            age_asset_bonus = 0.3 if age > 50 else 0.0
            if income < 30000:
                asset = "low"
            elif income < 80000 - age_asset_bonus * 50000:
                asset = "medium"
            else:
                asset = "high"

            records.append({
                "pid": i,
                "age": age,
                "gender": genders[i],
                "metro_status": metros[i],
                "state": states[i],
                "education": edu,
                "race": race,
                "political_leaning": parties[i],
                "income_annual": round(income, 2),
                "asset_level": asset,
                "debt_has_loan": has_loan,
                "debt_amount": round(debt_amount, 2),
                "homeownership": homeownership,
                "has_student_debt": has_student_debt,
                "has_health_insurance": has_insurance,
                "employ_status": employ,
                "job_satisfaction": round(job_sat, 1),
                "happiness": round(happiness, 1),
                "life_satisfaction": round(life_sat, 1),
                "future_confidence": round(future_conf, 1),
                "income_local_rank": round(inc_rank, 1),
                "social_status": round(social_stat, 1),
                "financial_satisfaction": round(fin_sat, 1),
                "work_reward_belief": round(work_belief, 1),
                "mobility_belief": round(mob_belief, 1),
                "health_status": health_status,
                "social_trust": social_trust,
            })

        pop = pd.DataFrame(records)

        # Compute income percentile
        pop["income_percentile"] = pop["income_annual"].rank(pct=True) * 100

        logger.info(f"US population built: {len(pop)} agents")
        for col in ["race", "political_leaning", "metro_status", "education"]:
            dist = pop[col].value_counts(normalize=True)
            logger.info(f"  {col}: {dict(dist.round(3))}")

        return pop
