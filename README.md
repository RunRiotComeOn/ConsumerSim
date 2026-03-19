# ConsumerSim

ConsumerSim is a U.S. consumer sentiment simulation project. The currently runnable entry point is `main_us.py`, and the goal is to generate Michigan-style consumer sentiment indices:

- `ICS` (`Index of Consumer Sentiment`)
- `ICC` (`Index of Current Conditions`)
- `ICE` (`Index of Consumer Expectations`)

The project combines a synthetic population, LLM inference over a subset of core agents, and Bayesian posterior expansion to the full population, then outputs indices, validation results, and plots.

## Current Entry Points

Main entry point:

```bash
python main_us.py
```

Diagnostic script:

```bash
python diagnose_llm.py --month 2025-03 --agents 100
```

## Project Structure

```text
ConsumerSim/
├─ config/
│  └─ config_us.yaml               # Active runtime configuration
├─ data/
│  ├─ cache_us/                    # Macro data / population cache
│  ├─ ces/                         # CES parameters and raw data
│  └─ gss/                         # GSS parameters and raw data
├─ deps/                           # Local vendored dependencies
├─ docs/
│  └─ profile_audit.md
├─ notebooks/                      # Research / experiment notebooks
├─ results_us/
│  ├─ monthly_backtest.csv         # Monthly backtest output
│  └─ figures/                     # Plot output directory
├─ src/
│  ├─ agents/
│  │  ├─ base.py                   # Shared agent types and response structures
│  │  └─ base_us.py                # USConsumerAgent
│  ├─ behavior_model/
│  │  ├─ demo_guidance_us.py       # Demographic priors / mapping rules
│  │  └─ predictor_us.py           # Bayesian posterior expansion
│  ├─ data/
│  │  ├─ macro_context.py          # Macro context assembly
│  │  └─ *_collector.py            # FRED / Census / news collectors, etc.
│  ├─ evaluation/
│  │  ├─ micro_us.py               # Micro-level validation
│  │  └─ structural_us.py          # Structural validation
│  ├─ llm/
│  │  ├─ inference_us.py           # OpenAI-compatible inference wrapper
│  │  ├─ prompts_us.py             # Prompt construction
│  │  └─ response_parser.py        # LLM response parsing
│  ├─ population/
│  │  └─ builder_us.py             # U.S. synthetic population builder
│  ├─ simulation/
│  │  ├─ engine_us.py              # Single-run simulation engine
│  │  ├─ monthly_runner.py         # Monthly / backtest / forecast runner
│  │  └─ index.py                  # ICS / ICC / ICE calculation
│  └─ visualization/
│     ├─ plots.py                  # Shared distribution / breakdown plots
│     └─ plots_us.py               # U.S.-specific plots
├─ diagnose_llm.py                 # Diagnose LLM output for a target month
├─ main_us.py                      # Main entry point
├─ README.md
└─ requirements.txt                # Python dependencies
```

## Execution Flow

The current codebase follows this workflow:

1. `USPopulationBuilder` constructs a synthetic U.S. population from ACS / CPS / GSS / CES and related sources.
2. `USSimulationEngine.build_agents()` creates agents and samples core agents using `core_agent_ratio`.
3. Unless `--no-llm` is used, the prior is calibrated with the latest or target Michigan `UMCSENT`.
4. Single-run mode fetches macro context; monthly mode builds a month-specific visible data window.
5. `USLLMInferenceEngine` runs LLM inference only for core agents.
6. `USBayesianPredictor` updates the posterior from core-agent responses and fills responses for the remaining agents.
7. `compute_indices()` calculates `ICS / ICC / ICE` and the component-level relative indices.
8. Single-run mode exports CSVs, validation outputs, and figures; monthly backtests write `monthly_backtest.csv`.

## Supported Run Modes

### 1. Single Simulation

```bash
python main_us.py
python main_us.py --agents 3000
python main_us.py --no-llm
python main_us.py --config config/config_us.yaml
```

Notes:

- Default mode is `--mode single`
- `--no-llm` skips LLM calls and uses `_inject_synthetic_responses()` to generate test responses
- Single-run mode outputs result tables, validation tables, and figures

### 2. Simulation for a Specific Month

```bash
python main_us.py --mode monthly --month 2025-06
python main_us.py --mode monthly --month 2025-06 --agents 1000
```

Notes:

- The main implementation lives in `src/simulation/monthly_runner.py`
- It constructs a survey window for the target month to avoid leaking future data
- It prioritizes the previous month's real `UMCSENT` as a prior anchor

### 3. Historical Backtest

```bash
python main_us.py --mode backtest --start 2024-06 --end 2026-01
```

Notes:

- Runs month by month
- Outputs `results_us/monthly_backtest.csv`
- If enough ground truth values are available, it reports correlation, `MAE`, and `RMSE`

### 4. Current-Month Forecast

```bash
python main_us.py --mode forecast
```

Notes:

- The current implementation calls `MonthlyRunner.forecast_next_month()`
- In practice, this runs a monthly simulation for the current month

## Diagnostic Script

`diagnose_llm.py` is used to inspect whether LLM inference for a given month looks reasonable:

```bash
python diagnose_llm.py --month 2025-03 --agents 100
```

It outputs:

- The macro context sent to the LLM
- The response distribution of core agents
- Core-agent `ICS`
- Final `ICS` after posterior expansion
- Ground-truth `ICS` for the target month, if available
- Example profiles of selected core agents

## Configuration

The active runtime configuration file is `config/config_us.yaml`, which mainly contains the following sections:

### `data`

- `processed_dir`: processed data directory
- `cache_dir`: cache directory
- `fred_api_key`: FRED key
- `census_api_key`: Census API key
- `news_api_key`: NewsAPI key
- `nyt_api_key`: New York Times Archive API key
- `guardian_api_key`: Guardian API key
- `bing_api_key`: Bing News Search key; supported by the code, though not filled by default in the current config
- `bing_endpoint`: Bing API endpoint
- `news_provider`: `nyt_archive | nyt | guardian | bing | newsapi`
- `nyt_us_only`, `guardian_us_only`: news filtering options
- `use_ustr_signal`, `use_gdelt_signal`, `use_news_signal`, `use_trends_signal`: whether to enable each macro signal
- `trends_geo`, `trends_max_batches`: Google Trends settings
- `authoritative_news_domains`: whitelist of allowed domains for NewsAPI

### `simulation`

- `total_agents`: total number of agents
- `core_agent_ratio`: share of core agents
- `batch_size`: number of agents per LLM inference batch
- `random_seed`: random seed
- `survey_window_cutoff_day`: survey window cutoff day
- `data_integrity_fail_on_violation`: whether monthly mode should fail immediately when data integrity checks fail

### `llm`

- `provider`: current default is `openai`
- `model`: current default is `gpt-4o-mini`
- `base_url`: OpenAI-compatible base URL; if empty, the default OpenAI endpoint is used
- `api_key`: can be written directly in the config
- `api_key_env`: can also be read from an environment variable; the current field is `OPENAI_API_KEY`
- `max_retries`, `temperature`, `timeout`, `max_tokens`

### `posterior`

- `prior_strength`: posterior prior strength

### `index`

- `icc_components`
- `ice_components`
- `ics_components`

### `output`

- `results_dir`
- `figures_dir`

## Output Files

By default, the current code writes outputs to `results_us/`:

- `results_us/us_simulation_results.csv`
- `results_us/us_micro_validation.csv`
- `results_us/us_structural_validation.csv`
- `results_us/monthly_backtest.csv`
- `results_us/figures/us_agent_profiles.png`
- `results_us/figures/us_response_distribution.png`
- `results_us/figures/us_demographic_breakdown.png`
- `results_us/figures/us_validation_summary.png`
- `results_us/figures/*.png`

## Data and Cache

The project currently uses or generates the following local data:

- `data/cache_us/*.parquet`
- `data/cache_us/*.json`
- `data/cache_us/*.xlsx`
- `data/ces/ces2022.csv`
- `data/ces/ces_party_params.json`
- `data/gss/2022/GSS2022.dta`
- `data/gss/gss_params.json`

Notes:

- Population caches are used to speed up synthetic population generation
- Each `collector` caches macro data and news summaries in `data/cache_us`
- `deps/` is a local dependency directory inside the repository; it is not part of the core source tree, but it is imported at runtime

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The main dependencies currently listed in `requirements.txt` include:

- `pandas`
- `numpy`
- `pyreadstat`
- `openpyxl`
- `pyyaml`
- `requests`
- `pytrends`
- `scikit-learn`
- `xgboost`
- `openai`
- `scipy`
- `pydantic`
- `matplotlib`

## Recommended Before Running

- Check whether the paths in `config/config_us.yaml` fit your machine
- If you do not want to store keys in the config file, prefer using environment variables
- For a first run, start with a smaller sample such as `--agents 100` or `--agents 500`
- If you only want to verify the pipeline end to end, start with `--no-llm`

## README Alignment Notes

This README is aligned to the code as it currently behaves, especially in the following ways:

- The project currently has only a U.S. entry point; there is no other country entry point
- The main executable script is `main_us.py`
- Monthly runs, backtests, and forecasts are all driven by `MonthlyRunner`
- The diagnostic entry point `diagnose_llm.py` is included
- Configuration field names are updated to match `config/config_us.yaml`
- The default `llm` values are updated to `openai / gpt-4o-mini`
- Output filenames are updated to match the actual export paths in the codebase
