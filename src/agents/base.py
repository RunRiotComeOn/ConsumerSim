"""
Base agent dataclass. Agents are lightweight — no Mesa overhead needed
since we're running batch LLM inference, not step-by-step agent logic.
"""
from dataclasses import dataclass, field
from typing import Optional


SURVEY_RESPONSES = ("better", "same", "worse", "good", "uncertain", "bad")


@dataclass
class SurveyResponse:
    """Five-question Michigan-style consumer survey response."""
    PAGO: str = "same"    # Personal finances vs last year
    DUR: str = "uncertain"  # Good time to buy durables
    PEXP: str = "same"    # Expected personal finances next year
    BUS12: str = "uncertain"  # Business conditions next 12 months
    BUS5: str = "uncertain"   # Business conditions next 5 years

    def is_valid(self) -> bool:
        pago_ok = self.PAGO in ("better", "same", "worse")
        dur_ok = self.DUR in ("good", "uncertain", "bad")
        pexp_ok = self.PEXP in ("better", "same", "worse")
        bus12_ok = self.BUS12 in ("good", "uncertain", "bad")
        bus5_ok = self.BUS5 in ("good", "uncertain", "bad")
        return all([pago_ok, dur_ok, pexp_ok, bus12_ok, bus5_ok])

    def to_dict(self) -> dict:
        return {
            "PAGO": self.PAGO,
            "DUR": self.DUR,
            "PEXP": self.PEXP,
            "BUS12": self.BUS12,
            "BUS5": self.BUS5,
        }


@dataclass
class ConsumerAgent:
    """
    A simulated Chinese consumer agent.

    Profile attributes come from CFPS 2018 / CHFS 2019 survey data.
    All subjective fields are real survey measurements, NOT synthetic.
    """
    agent_id: int
    pid: Optional[float] = None

    # --- Demographics (objective, from CFPS) ---
    age: float = 35.0
    gender: str = "male"          # "male" / "female"
    urban: str = "urban"          # "urban" / "rural"
    province: str = "Guangdong"
    region: str = "East"
    education: str = "college"

    # --- Financial (objective, from CFPS + famecon) ---
    income_annual: float = 50000.0
    income_percentile: float = 50.0
    asset_level: str = "medium"   # "low" / "medium" / "high"  (from ft1 savings)
    debt_has_loan: int = 0        # 0/1 whether has bank loan (from ft5)
    debt_amount: float = 0.0      # outstanding loan amount (from ft501)

    # --- Employment (real, from CFPS) ---
    employ_status: str = "employed"  # "employed" / "unemployed" / "retired" / "other"
    job_satisfaction: float = 3.0    # 1-5, from qg406 (NaN → 3.0 default for non-workers)

    # --- Subjective perception (ALL real CFPS measurements) ---
    happiness: float = 7.5           # 0-10, from qm2016 "您有多幸福"
    life_satisfaction: float = 4.0   # 1-5, from qn12012 "对自己生活满意度"
    future_confidence: float = 4.0   # 1-5, from qn12016 "对自己未来信心程度"
    income_local_rank: float = 3.0   # 1-5, from qn8011 "您的收入在本地"
    social_status: float = 3.0       # 1-5, from qn8012 "您的地位"
    job_income_satisfaction: float = 3.0  # 1-5, from qg401
    work_reward_belief: float = 3.0  # 1-4, from wv104 "努力工作能有回报"
    mobility_belief: float = 3.0     # 1-4, from wv108 "提高生活水平机会很大"
    health_change: str = "same"      # "better" / "same" / "worse", from qp202
    social_trust: str = "trust"      # "trust" / "cautious", from qn1001

    # Agent type
    is_core: bool = False

    # Survey response (set after simulation step)
    response: SurveyResponse = field(default_factory=SurveyResponse)

    def to_profile_text(self) -> str:
        """Render agent profile as text for LLM prompt."""
        debt_str = f"has bank loan (¥{self.debt_amount:,.0f})" if self.debt_has_loan else "no bank loan"
        return (
            f"Age: {int(self.age)}\n"
            f"Gender: {self.gender}\n"
            f"Region: {self.province} ({self.region} China)\n"
            f"Urban/Rural: {self.urban}\n"
            f"Education: {self.education}\n"
            f"Employment: {self.employ_status}\n"
            f"Annual income: ¥{self.income_annual:,.0f} (percentile: {self.income_percentile:.0f})\n"
            f"Assets: {self.asset_level} | Debt: {debt_str}\n"
            f"Self-rated happiness: {self.happiness:.0f}/10\n"
            f"Life satisfaction: {self.life_satisfaction:.0f}/5\n"
            f"Confidence in own future: {self.future_confidence:.0f}/5\n"
            f"Income rank in local area: {self.income_local_rank:.0f}/5\n"
            f"Self-rated social status: {self.social_status:.0f}/5\n"
            f"Job satisfaction: {self.job_satisfaction:.0f}/5\n"
            f"Job income satisfaction: {self.job_income_satisfaction:.0f}/5\n"
            f"Believes hard work pays off: {self.work_reward_belief:.0f}/4\n"
            f"Believes opportunity to improve living standard: {self.mobility_belief:.0f}/4\n"
            f"Health change vs last year: {self.health_change}\n"
            f"Social trust: {self.social_trust}"
        )

    def to_feature_vector(self) -> dict:
        """Return numeric features for ML behavior model."""
        return {
            "age": self.age,
            "gender_female": 1 if self.gender == "female" else 0,
            "urban": 1 if self.urban == "urban" else 0,
            "income_percentile": self.income_percentile,
            # Asset / debt
            "asset_low": 1 if self.asset_level == "low" else 0,
            "asset_medium": 1 if self.asset_level == "medium" else 0,
            "asset_high": 1 if self.asset_level == "high" else 0,
            "debt_has_loan": self.debt_has_loan,
            "debt_amount_log": _safe_log(self.debt_amount),
            # Employment
            "employed": 1 if self.employ_status == "employed" else 0,
            "retired": 1 if self.employ_status == "retired" else 0,
            "job_satisfaction": self.job_satisfaction,
            # Subjective perception (all real survey data)
            "happiness": self.happiness,
            "life_satisfaction": self.life_satisfaction,
            "future_confidence": self.future_confidence,
            "income_local_rank": self.income_local_rank,
            "social_status": self.social_status,
            "job_income_satisfaction": self.job_income_satisfaction,
            "work_reward_belief": self.work_reward_belief,
            "mobility_belief": self.mobility_belief,
            "health_better": 1 if self.health_change == "better" else 0,
            "health_worse": 1 if self.health_change == "worse" else 0,
            "social_trust": 1 if self.social_trust == "trust" else 0,
            # Education
            "edu_primary": 1 if self.education == "primary" else 0,
            "edu_middle": 1 if self.education == "middle" else 0,
            "edu_high": 1 if self.education == "high" else 0,
            "edu_college": 1 if self.education in ("college", "grad") else 0,
            # Region
            "region_east": 1 if self.region == "East" else 0,
            "region_central": 1 if self.region == "Central" else 0,
            "region_west": 1 if self.region == "West" else 0,
        }

    @classmethod
    def from_row(cls, row: dict, agent_id: int, is_core: bool = False) -> "ConsumerAgent":
        """Construct agent from a population DataFrame row."""
        return cls(
            agent_id=agent_id,
            pid=row.get("pid"),
            age=float(row.get("age", 35)),
            gender=str(row.get("gender_label", "male")),
            urban=str(row.get("urban_label", "urban")),
            province=str(row.get("province", "Guangdong")),
            region=str(row.get("region", "East")),
            education=str(row.get("education", "college")),
            income_annual=float(row.get("effective_income", 50000)),
            income_percentile=float(row.get("income_percentile", 50)),
            asset_level=str(row.get("asset_level", "medium")),
            debt_has_loan=int(row.get("debt_has_loan", 0)),
            debt_amount=float(row.get("debt_amount", 0)),
            employ_status=str(row.get("employ_status", "employed")),
            job_satisfaction=float(row.get("job_satisfaction", 3.0)),
            happiness=float(row.get("happiness", 7.5)),
            life_satisfaction=float(row.get("life_satisfaction", 4.0)),
            future_confidence=float(row.get("future_confidence", 4.0)),
            income_local_rank=float(row.get("income_local_rank", 3.0)),
            social_status=float(row.get("social_status", 3.0)),
            job_income_satisfaction=float(row.get("job_income_satisfaction", 3.0)),
            work_reward_belief=float(row.get("work_reward_belief", 3.0)),
            mobility_belief=float(row.get("mobility_belief", 3.0)),
            health_change=str(row.get("health_change", "same")),
            social_trust=str(row.get("social_trust", "trust")),
            is_core=is_core,
        )


def _safe_log(x: float) -> float:
    """Log transform for monetary amounts, handling zero."""
    import math
    return math.log1p(max(0, x))
