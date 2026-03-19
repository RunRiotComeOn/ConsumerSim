"""
Monthly Simulation Runner.

Runs the US ConsumerSim for each month, using that month's real FRED data,
and compares simulated ICS against the real Michigan ICS (UMCSENT).

This enables:
  1. Monthly forecasting: predict next month's ICS before release
  2. Backtesting: run historical months and validate against real ICS
  3. Tracking: monitor simulation accuracy over time
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonthlyRunner:
    """Run monthly ICS simulations and compare with real Michigan data."""

    def __init__(self, config: dict):
        self.config = config
        self.fred_api_key = config["data"].get("fred_api_key", "")
        self._fred = None
        self._sce = None
        self._ustr = None
        self._gdelt = None
        self._news = None
        self._trends = None

    def _get_fred(self):
        if self._fred is None:
            from src.data.fred_collector import FREDCollector
            self._fred = FREDCollector(self.fred_api_key)
        return self._fred

    def _get_sce(self):
        if self._sce is None:
            from src.data.sce_collector import SCECollector
            cache_dir = self.config["data"].get("cache_dir")
            self._sce = SCECollector(cache_dir=cache_dir)
        return self._sce

    def _get_ustr(self):
        if self._ustr is None:
            from src.data.ustr_collector import USTRCollector
            cache_dir = self.config["data"].get("cache_dir")
            self._ustr = USTRCollector(
                cache_dir=cache_dir,
                survey_start_day=self._survey_start_day(),
            )
        return self._ustr

    def _get_gdelt(self):
        if self._gdelt is None:
            from src.data.gdelt_collector import GDELTCollector
            cache_dir = self.config["data"].get("cache_dir")
            self._gdelt = GDELTCollector(
                cache_dir=cache_dir,
                survey_start_day=self._survey_start_day(),
            )
        return self._gdelt

    def _get_news(self):
        if self._news is None:
            data_cfg = self.config.get("data", {})
            cache_dir = data_cfg.get("cache_dir")
            provider = data_cfg.get("news_provider", "nyt_archive").lower()
            if provider == "newsapi":
                from src.data.news_api_collector import NewsAPICollector, DEFAULT_DOMAINS
                api_key = data_cfg.get("news_api_key", "")
                domains = data_cfg.get("authoritative_news_domains") or DEFAULT_DOMAINS
                self._news = NewsAPICollector(api_key=api_key, cache_dir=cache_dir, domains=domains)
            elif provider == "bing":
                from src.data.bing_news_collector import BingNewsCollector, DEFAULT_BING_ENDPOINT
                api_key = data_cfg.get("bing_api_key", "")
                endpoint = data_cfg.get("bing_endpoint", DEFAULT_BING_ENDPOINT)
                self._news = BingNewsCollector(api_key=api_key, cache_dir=cache_dir, endpoint=endpoint)
            elif provider in {"nyt", "nyt_archive"}:
                from src.data.nyt_archive_collector import NYTArchiveCollector
                api_key = data_cfg.get("nyt_api_key", "")
                us_only = data_cfg.get("nyt_us_only", True)
                self._news = NYTArchiveCollector(api_key=api_key, cache_dir=cache_dir, us_only=us_only)
            else:
                from src.data.guardian_collector import GuardianCollector
                api_key = data_cfg.get("guardian_api_key", "")
                us_only = data_cfg.get("guardian_us_only", True)
                self._news = GuardianCollector(api_key=api_key, cache_dir=cache_dir, us_only=us_only)
        return self._news

    def _get_trends(self):
        if self._trends is None:
            from src.data.google_trends_collector import GoogleTrendsCollector
            data_cfg = self.config.get("data", {})
            cache_dir = data_cfg.get("cache_dir")
            geo = data_cfg.get("trends_geo", "US")
            max_batches = data_cfg.get("trends_max_batches", 1)
            self._trends = GoogleTrendsCollector(cache_dir=cache_dir, geo=geo, max_batches=max_batches)
        return self._trends

    def _survey_start_day(self) -> int:
        sim_cfg = self.config.get("simulation", {})
        day = sim_cfg.get("survey_window_cutoff_day", sim_cfg.get("survey_second_week_start_day", 25))
        return max(1, min(int(day), 28))

    def _second_week_start(self, year: int, month: int) -> datetime:
        return datetime(year, month, self._survey_start_day())

    @staticmethod
    def _prev_month(year: int, month: int) -> Tuple[int, int]:
        if month == 1:
            return year - 1, 12
        return year, month - 1

    def _survey_cutoff(self, year: int, month: int) -> datetime:
        return self._second_week_start(year, month)

    def _survey_window(self, year: int, month: int) -> Tuple[str, str]:
        """Window: previous month cutoff day -> current month cutoff day (inclusive)."""
        py, pm = self._prev_month(year, month)
        start_dt = datetime(py, pm, self._survey_start_day()).strftime("%Y-%m-%d")
        end_dt = datetime(year, month, self._survey_start_day()).strftime("%Y-%m-%d")
        return start_dt, end_dt

    @staticmethod
    def _select_asof_value(data: pd.Series, asof_date: datetime) -> Optional[Tuple[datetime, float]]:
        valid = data.dropna()
        if len(valid) == 0:
            return None
        eligible = valid[valid.index <= asof_date]
        if len(eligible) == 0:
            return None
        dt = eligible.index[-1]
        return dt.to_pydatetime(), float(eligible.iloc[-1])

    def get_real_ics_history(self, start: str = "2024-01-01",
                              end: str = None) -> pd.Series:
        """Fetch real Michigan ICS monthly history from FRED."""
        fred = self._get_fred()
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        data = fred.fetch_series("UMCSENT", start=start, end=end)
        logger.info(f"Real ICS history: {len(data)} months ({data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')})")
        return data

    def get_macro_for_month(self, year: int, month: int) -> Dict[str, object]:
        """Get macro indicators as of a specific month.

        Survey timing rule:
        - Inputs are cut off at the configured day of month t (default day 25).
        - Inputs should only use information available strictly before that point.
        """
        fred = self._get_fred()
        cutoff_dt = self._survey_cutoff(year, month)
        end_date = cutoff_dt.strftime("%Y-%m-%d")

        # Also need year-ago data for YoY
        start_yoy = datetime(year - 1, month, 1).strftime("%Y-%m-%d")

        key_series = ["UNRATE", "CPIAUCSL", "GASREGW", "MORTGAGE30US",
                      "FEDFUNDS", "MICH", "SP500", "VIXCLS", "CSUSHPINSA", "PAYEMS"]
        # Monthly releases can lag and are often unavailable for month t by the second week.
        monthly_series = {"UNRATE", "CPIAUCSL", "FEDFUNDS", "MICH", "CSUSHPINSA", "PAYEMS"}

        snapshot = {}
        yoy = {}
        for sid in key_series:
            try:
                data = fred.fetch_series(sid, start=start_yoy, end=end_date)
                data = data.dropna()
                if len(data) == 0:
                    continue

                # For monthly series, force t-1 vintage to avoid month-t lookahead.
                if sid in monthly_series:
                    py, pm = self._prev_month(year, month)
                    asof_dt = datetime(py, pm, 28)
                else:
                    asof_dt = cutoff_dt

                selected = self._select_asof_value(data, asof_dt)
                if selected is None:
                    continue
                obs_dt, obs_val = selected
                snapshot[sid] = obs_val

                # Find value closest to 12 months before selected observation.
                target = obs_dt - timedelta(days=365)
                idx = data.index.searchsorted(target)
                idx = max(0, min(idx, len(data) - 1))
                if idx > 0:
                    d_left = abs((data.index[idx - 1] - target).days)
                    d_right = abs((data.index[idx] - target).days)
                    if d_left < d_right:
                        idx = idx - 1
                ago_val = float(data.iloc[idx])
                if ago_val != 0 and abs((data.index[idx] - target).days) < 45:
                    yoy[sid] = ((obs_val - ago_val) / abs(ago_val)) * 100
            except Exception as e:
                logger.debug(f"  Skip {sid} for {year}-{month}: {e}")

        sce_snapshot = {}
        try:
            py, pm = self._prev_month(year, month)
            sce_snapshot = self._get_sce().get_month_snapshot(py, pm)
        except Exception as e:
            logger.debug(f"  Skip SCE for {year}-{month}: {e}")

        ustr_summary = {}
        if self.config.get("data", {}).get("use_ustr_signal", False):
            try:
                ustr_summary = self._get_ustr().get_month_summary(year, month)
                logger.info(
                    f"USTR window {year}-{month:02d}: "
                    f"events={ustr_summary.get('ustr_trade_event_count', 0)}, "
                    f"tariff={ustr_summary.get('ustr_tariff_event_count', 0)}"
                )
            except Exception as e:
                logger.warning(f"Skip USTR for {year}-{month}: {e}")

        gdelt_summary = {}
        if self.config.get("data", {}).get("use_gdelt_signal", False):
            try:
                gdelt_summary = self._get_gdelt().get_month_summary(year, month)
                logger.info(
                    f"GDELT window {year}-{month:02d}: "
                    f"econ_volume={gdelt_summary.get('gdelt_econ_news_volume', 0.0):.0f}"
                )
            except Exception as e:
                logger.warning(f"Skip GDELT for {year}-{month}: {e}")

        news_summary = {}
        if self.config.get("data", {}).get("use_news_signal", self.config.get("data", {}).get("use_newsapi_signal", True)):
            try:
                start_dt, end_dt = self._survey_window(year, month)
                news_summary = self._get_news().get_window_summary(start_dt, end_dt, limit=8)
                provider = self.config.get("data", {}).get("news_provider", "nyt_archive").lower()
                err = news_summary.get("error")
                logger.info(
                    f"{provider} news window {start_dt}..{end_dt}: "
                    f"events={news_summary.get('news_event_count', 0)}, "
                    f"trade_policy={news_summary.get('tariff_event_count', 0)}"
                    + (f", error={err}" if err else "")
                )
            except Exception as e:
                logger.warning(f"Skip news collector for {year}-{month}: {e}")

        trends_summary = {}
        if self.config.get("data", {}).get("use_trends_signal", True):
            try:
                start_dt, end_dt = self._survey_window(year, month)
                trends_summary = self._get_trends().get_window_summary(start_dt, end_dt, limit=6)
                logger.info(
                    f"GoogleTrends window {start_dt}..{end_dt}: "
                    f"attention={trends_summary.get('trends_attention_score', 0.0):.1f}, "
                    f"status={trends_summary.get('status', 'unknown')}"
                )
            except Exception as e:
                logger.warning(f"Skip GoogleTrends for {year}-{month}: {e}")

        return {
            "snapshot": snapshot,
            "yoy": yoy,
            "sce": sce_snapshot,
            "ustr": ustr_summary,
            "gdelt": gdelt_summary,
            "newsapi": news_summary,
            "trends": trends_summary,
        }

    def build_month_context(self, year: int, month: int) -> str:
        """Build macro context string for a specific month."""
        macro = self.get_macro_for_month(year, month)
        return self._render_month_context(year, month, macro)

    def _render_month_context(self, year: int, month: int, macro: Dict[str, object]) -> str:
        """Render context text from an already-fetched macro payload."""
        snap = macro["snapshot"]
        yoy = macro["yoy"]
        sce = macro.get("sce", {})
        ustr = macro.get("ustr", {})
        gdelt = macro.get("gdelt", {})
        news = macro.get("newsapi", {})
        trends = macro.get("trends", {})

        def _yoy(sid):
            if sid in yoy:
                v = yoy[sid]
                return f" ({'+' if v > 0 else ''}{v:.1f}% YoY)"
            return ""

        parts = []
        if "UNRATE" in snap:
            parts.append(f"Unemployment rate: {snap['UNRATE']:.1f}%{_yoy('UNRATE')}")
        if "CPIAUCSL" in yoy:
            parts.append(f"Inflation (CPI YoY): {yoy['CPIAUCSL']:.1f}%")
        if "GASREGW" in snap:
            parts.append(f"Gas price: ${snap['GASREGW']:.2f}/gallon{_yoy('GASREGW')}")
        if "MORTGAGE30US" in snap:
            parts.append(f"30-year mortgage rate: {snap['MORTGAGE30US']:.2f}%{_yoy('MORTGAGE30US')}")
        if "FEDFUNDS" in snap:
            parts.append(f"Fed funds rate: {snap['FEDFUNDS']:.2f}%")
        if "MICH" in snap:
            parts.append(f"Inflation expectations: {snap['MICH']:.1f}%")
        if "SP500" in snap:
            parts.append(f"S&P 500: {snap['SP500']:,.0f}{_yoy('SP500')}")
        if "VIXCLS" in snap:
            parts.append(f"VIX volatility index: {snap['VIXCLS']:.1f}")
        if "CSUSHPINSA" in snap:
            parts.append(f"Home prices (Case-Shiller): {snap['CSUSHPINSA']:.1f}{_yoy('CSUSHPINSA')}")
        if "sce_inflation_1y" in sce:
            parts.append(f"NY Fed SCE 1-year inflation expectations: {sce['sce_inflation_1y']:.1f}%")
        if "sce_unemployment_expectation" in sce:
            parts.append(
                "NY Fed SCE probability unemployment will be higher in 1 year: "
                f"{sce['sce_unemployment_expectation']:.1f}%"
            )
        if "sce_job_loss_expectation" in sce:
            parts.append(f"NY Fed SCE perceived job loss probability: {sce['sce_job_loss_expectation']:.1f}%")
        if "sce_job_find_expectation" in sce:
            parts.append(f"NY Fed SCE perceived job finding probability: {sce['sce_job_find_expectation']:.1f}%")
        if self.config.get("data", {}).get("use_ustr_signal", False):
            parts.append(
                f"USTR trade-policy press releases in window: {ustr.get('ustr_trade_event_count', 0)} "
                f"(tariff-specific: {ustr.get('ustr_tariff_event_count', 0)})"
            )
        if self.config.get("data", {}).get("use_gdelt_signal", False):
            parts.append(
                f"GDELT economic shock news volume in window: {gdelt.get('gdelt_econ_news_volume', 0.0):.0f}"
            )
        if news.get("top_headlines"):
            parts.append("Authoritative news highlights in survey window:")
        for h in news.get("top_headlines", [])[:8]:
            title = h.get("title", "")
            source = h.get("source", "")
            date = h.get("date", "")
            topic = h.get("topic", "")
            tone = h.get("tone", "")
            summary = h.get("summary", "")
            if title:
                tags = ", ".join([t for t in (topic, tone) if t])
                if topic:
                    line = f"Authoritative headline [{tags}] ({source}, {date}): {title}"
                else:
                    line = f"Authoritative headline ({source}, {date}): {title}"
                if summary:
                    line += f" -- {summary}"
                parts.append(line)
        if self.config.get("data", {}).get("use_trends_signal", True):
            parts.append(
                f"Google Trends macro attention score (US): {trends.get('trends_attention_score', 0.0):.1f} "
                f"(status: {trends.get('status', 'unknown')})"
            )
            for kw in trends.get("top_keywords", [])[:4]:
                parts.append(f"Google Trends keyword: {kw.get('keyword')} ({kw.get('score', 0):.1f})")
        for article in gdelt.get("gdelt_headlines", [])[:2]:
            title = article.get("title")
            if title:
                parts.append(f"GDELT headline: {title}")
        for example in ustr.get("ustr_examples", [])[:2]:
            title = example.get("title")
            if title:
                parts.append(f"USTR example: {title}")

        header = f"US Economic Conditions as of {year}-{month:02d}:"
        return header + "\n" + "\n".join(f"  - {p}" for p in parts)

    def _data_integrity_violations(self, macro: Dict[str, object], year: int, month: int) -> List[str]:
        violations: List[str] = []
        snapshot = macro.get("snapshot", {}) or {}
        news = macro.get("newsapi", {}) or {}

        if len(snapshot) == 0:
            violations.append(f"{year}-{month:02d}: FRED snapshot is empty")

        if self.config.get("data", {}).get("use_news_signal", True):
            if news.get("news_event_count", 0) == 0:
                violations.append(f"{year}-{month:02d}: authoritative news event count is 0")

        return violations

    def get_prior_month_ics(self, year: int, month: int) -> Optional[float]:
        """Get the real ICS from the month BEFORE the target month.

        This is used as the prior anchor: we assume last month's sentiment
        is the baseline, and this month's LLM responses adjust from there.
        """
        # Previous month
        if month == 1:
            py, pm = year - 1, 12
        else:
            py, pm = year, month - 1

        try:
            fred = self._get_fred()
            start = f"{py}-{pm:02d}-01"
            end = f"{py}-{pm:02d}-28"
            data = fred.fetch_series("UMCSENT", start=start, end=end)
            if len(data) > 0:
                val = float(data.iloc[-1])
                logger.info(f"Prior month ICS ({py}-{pm:02d}): {val:.1f}")
                return val
        except Exception as e:
            logger.debug(f"Could not fetch prior month ICS: {e}")

        # Fallback: try to get any recent UMCSENT
        try:
            fred = self._get_fred()
            start = f"{year - 1}-01-01"
            end = f"{year}-{month:02d}-01"
            data = fred.fetch_series("UMCSENT", start=start, end=end)
            data = data.dropna()
            if len(data) > 0:
                val = float(data.iloc[-1])
                logger.info(f"Using most recent available ICS: {val:.1f} "
                           f"({data.index[-1].strftime('%Y-%m')})")
                return val
        except Exception:
            pass

        return None

    def run_single_month(self, year: int, month: int,
                          n_agents: int = 2000) -> Dict:
        """Run simulation for a single month with that month's macro data."""
        from src.simulation.engine_us import USSimulationEngine

        logger.info(f"\n{'='*60}")
        logger.info(f"Running simulation for {year}-{month:02d}")
        logger.info(f"{'='*60}")

        # Build macro payload and enforce data-integrity gate before inference.
        macro = self.get_macro_for_month(year, month)
        violations = self._data_integrity_violations(macro, year, month)
        if violations:
            for item in violations:
                logger.warning(f"Data integrity gate: {item}")
            if self.config.get("simulation", {}).get("data_integrity_fail_on_violation", True):
                raise RuntimeError(
                    "Data integrity gate failed; result is not trustworthy. "
                    + "; ".join(violations)
                )

        context = self._render_month_context(year, month, macro)
        logger.info(f"Macro context:\n{context}")

        # Override agent count
        config = dict(self.config)
        config["simulation"] = dict(config["simulation"])
        config["simulation"]["total_agents"] = n_agents
        config["simulation"]["survey_year"] = year
        config["simulation"]["survey_month"] = month

        engine = USSimulationEngine(config)
        engine.prediction_target_month = f"{year}-{month:02d}"
        engine.survey_cutoff_date = self._survey_cutoff(year, month).strftime("%Y-%m-%d")

        # Calibrate priors to previous month's real ICS
        prior_ics = self.get_prior_month_ics(year, month)
        if prior_ics is not None:
            engine.calibrate_to_real_ics(target_ics=prior_ics)
        else:
            logger.warning("No prior month ICS available, fetching latest UMCSENT")
            engine.calibrate_to_real_ics()

        engine.build_agents()
        engine.macro_context = context
        engine.run_llm_inference()
        engine.update_and_predict()
        indices = engine.compute_step_indices()

        # Get real ICS for this month
        real_ics = None
        try:
            fred = self._get_fred()
            start = f"{year}-{month:02d}-01"
            end = f"{year}-{month:02d}-28"
            real_data = fred.fetch_series("UMCSENT", start=start, end=end)
            if len(real_data) > 0:
                real_ics = float(real_data.iloc[-1])
        except Exception:
            pass

        result = {
            "year": year,
            "month": month,
            "sim_ICS": indices["ICS"],
            "sim_ICC": indices["ICC"],
            "sim_ICE": indices["ICE"],
            "real_ICS": real_ics,
        }

        if real_ics is not None:
            error = indices["ICS"] - real_ics
            result["error"] = error
            result["pct_error"] = (error / real_ics) * 100
            logger.info(f"  Simulated ICS: {indices['ICS']:.1f}")
            logger.info(f"  Real ICS:      {real_ics:.1f}")
            logger.info(f"  Error:         {error:+.1f} ({result['pct_error']:+.1f}%)")
        else:
            logger.info(f"  Simulated ICS: {indices['ICS']:.1f}")
            logger.info(f"  Real ICS:      not available")

        return result

    def run_backtest(self, start_year: int = 2024, start_month: int = 6,
                      end_year: int = 2026, end_month: int = 1,
                      n_agents: int = 2000) -> pd.DataFrame:
        """Run backtest across multiple months."""
        results = []
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)

        while current <= end:
            try:
                result = self.run_single_month(
                    current.year, current.month, n_agents=n_agents
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed for {current.year}-{current.month:02d}: {e}")

            # Next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        df = pd.DataFrame(results)

        if "real_ICS" in df.columns and "sim_ICS" in df.columns:
            valid = df.dropna(subset=["real_ICS", "sim_ICS"])
            if len(valid) > 1:
                from scipy import stats
                r, p = stats.pearsonr(valid["real_ICS"], valid["sim_ICS"])
                mae = (valid["sim_ICS"] - valid["real_ICS"]).abs().mean()
                rmse = np.sqrt(((valid["sim_ICS"] - valid["real_ICS"])**2).mean())
                logger.info(f"\n{'='*60}")
                logger.info(f"BACKTEST SUMMARY ({len(valid)} months)")
                logger.info(f"  Correlation (r): {r:.3f} (p={p:.4f})")
                logger.info(f"  MAE:  {mae:.1f}")
                logger.info(f"  RMSE: {rmse:.1f}")
                logger.info(f"  Mean real ICS:  {valid['real_ICS'].mean():.1f}")
                logger.info(f"  Mean sim ICS:   {valid['sim_ICS'].mean():.1f}")
                logger.info(f"{'='*60}")

        # Save results
        results_dir = self.config.get("output", {}).get("results_dir", "results_us")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "monthly_backtest.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Backtest results saved to {out_path}")

        return df

    def forecast_next_month(self, n_agents: int = 2000) -> Dict:
        """Forecast next month's ICS using latest available data."""
        now = datetime.now()
        # Predict for current month (data lag means we use last month's data)
        return self.run_single_month(now.year, now.month, n_agents=n_agents)
