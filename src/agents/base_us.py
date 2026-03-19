"""
Base agent dataclass for US version.
Adds race, political_leaning, homeownership, state, metro_status fields.
"""
from dataclasses import dataclass, field
from typing import Optional

from src.agents.base import SurveyResponse, SURVEY_RESPONSES, _safe_log

CENSUS_REGIONS = {
    "Northeast": [
        "Connecticut", "Maine", "Massachusetts", "New Hampshire",
        "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania",
    ],
    "Midwest": [
        "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
        "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin",
    ],
    "South": [
        "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky",
        "Louisiana", "Maryland", "Mississippi", "North Carolina", "Oklahoma",
        "South Carolina", "Tennessee", "Texas", "Virginia", "West Virginia",
        "District of Columbia",
    ],
    "West": [
        "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho",
        "Montana", "Nevada", "New Mexico", "Oregon", "Utah", "Washington", "Wyoming",
    ],
}

STATE_TO_REGION = {}
for region, states in CENSUS_REGIONS.items():
    for state in states:
        STATE_TO_REGION[state] = region


@dataclass
class USConsumerAgent:
    """A simulated US consumer agent with race, party, homeownership dimensions."""
    agent_id: int
    pid: Optional[float] = None

    age: float = 35.0
    gender: str = "male"
    metro_status: str = "metro"
    state: str = "California"
    region: str = "West"
    education: str = "some_college"
    race: str = "White"
    political_leaning: str = "Independent"

    income_annual: float = 55000.0
    income_percentile: float = 50.0
    asset_level: str = "medium"
    debt_has_loan: int = 0
    debt_amount: float = 0.0
    homeownership: str = "renter"
    has_student_debt: int = 0
    has_health_insurance: int = 1

    employ_status: str = "employed"
    job_satisfaction: float = 3.0

    happiness: float = 2.0
    life_satisfaction: float = 4.0
    future_confidence: float = 3.0
    income_local_rank: float = 3.0
    social_status: float = 3.0
    financial_satisfaction: float = 3.0
    work_reward_belief: float = 3.0
    mobility_belief: float = 3.0
    health_status: str = "good"
    social_trust: str = "cautious"

    is_core: bool = False
    response: SurveyResponse = field(default_factory=SurveyResponse)

    def to_profile_text(self) -> str:
        debt_parts = []
        if self.debt_has_loan:
            debt_parts.append(f"mortgage/loan (${self.debt_amount:,.0f})")
        if self.has_student_debt:
            debt_parts.append("student loan debt")
        debt_str = ", ".join(debt_parts) if debt_parts else "no major debt"
        insurance_str = "has health insurance" if self.has_health_insurance else "no health insurance"

        return (
            f"Age: {int(self.age)}\n"
            f"Gender: {self.gender}\n"
            f"Race/Ethnicity: {self.race}\n"
            f"State: {self.state} ({self.region} US)\n"
            f"Metro status: {self.metro_status}\n"
            f"Education: {self.education}\n"
            f"Political leaning: {self.political_leaning}\n"
            f"Employment: {self.employ_status}\n"
            f"Annual income: ${self.income_annual:,.0f} (percentile: {self.income_percentile:.0f})\n"
            f"Assets: {self.asset_level} | Debt: {debt_str}\n"
            f"Homeownership: {self.homeownership} | {insurance_str}\n"
            f"Self-rated happiness: {self.happiness:.0f}/3\n"
            f"Life satisfaction: {self.life_satisfaction:.0f}/5\n"
            f"Confidence in financial future: {self.future_confidence:.0f}/5\n"
            f"Income rank vs American families: {self.income_local_rank:.0f}/5\n"
            f"Subjective social class: {self.social_status:.0f}/5\n"
            f"Job satisfaction: {self.job_satisfaction:.0f}/4\n"
            f"Financial satisfaction: {self.financial_satisfaction:.0f}/5\n"
            f"Believes hard work leads to success: {self.work_reward_belief:.0f}/4\n"
            f"Believes in opportunity to get ahead: {self.mobility_belief:.0f}/4\n"
            f"Health status: {self.health_status}\n"
            f"Social trust: {self.social_trust}"
        )

    def to_feature_vector(self) -> dict:
        return {
            "age": self.age,
            "gender_female": 1 if self.gender == "female" else 0,
            "metro": 1 if self.metro_status == "metro" else 0,
            "income_percentile": self.income_percentile,
            "race_white": 1 if self.race == "White" else 0,
            "race_black": 1 if self.race == "Black" else 0,
            "race_hispanic": 1 if self.race == "Hispanic" else 0,
            "race_asian": 1 if self.race == "Asian" else 0,
            "party_democrat": 1 if self.political_leaning == "Democrat" else 0,
            "party_republican": 1 if self.political_leaning == "Republican" else 0,
            "asset_low": 1 if self.asset_level == "low" else 0,
            "asset_medium": 1 if self.asset_level == "medium" else 0,
            "asset_high": 1 if self.asset_level == "high" else 0,
            "debt_has_loan": self.debt_has_loan,
            "debt_amount_log": _safe_log(self.debt_amount),
            "homeowner": 1 if self.homeownership == "owner" else 0,
            "has_student_debt": self.has_student_debt,
            "has_health_insurance": self.has_health_insurance,
            "employed": 1 if self.employ_status == "employed" else 0,
            "retired": 1 if self.employ_status == "retired" else 0,
            "job_satisfaction": self.job_satisfaction,
            "happiness": self.happiness,
            "life_satisfaction": self.life_satisfaction,
            "future_confidence": self.future_confidence,
            "income_local_rank": self.income_local_rank,
            "social_status": self.social_status,
            "financial_satisfaction": self.financial_satisfaction,
            "work_reward_belief": self.work_reward_belief,
            "mobility_belief": self.mobility_belief,
            "health_excellent": 1 if self.health_status == "excellent" else 0,
            "health_good": 1 if self.health_status == "good" else 0,
            "health_fair": 1 if self.health_status == "fair" else 0,
            "health_poor": 1 if self.health_status == "poor" else 0,
            "social_trust": 1 if self.social_trust == "trust" else 0,
            "edu_less_hs": 1 if self.education == "less_than_hs" else 0,
            "edu_hs": 1 if self.education == "high_school" else 0,
            "edu_some_college": 1 if self.education == "some_college" else 0,
            "edu_bachelors": 1 if self.education == "bachelors" else 0,
            "edu_graduate": 1 if self.education == "graduate" else 0,
            "region_northeast": 1 if self.region == "Northeast" else 0,
            "region_midwest": 1 if self.region == "Midwest" else 0,
            "region_south": 1 if self.region == "South" else 0,
            "region_west": 1 if self.region == "West" else 0,
        }

    @classmethod
    def from_row(cls, row: dict, agent_id: int, is_core: bool = False) -> "USConsumerAgent":
        state = str(row.get("state", "California"))
        region = STATE_TO_REGION.get(state, "West")
        return cls(
            agent_id=agent_id, pid=row.get("pid"),
            age=float(row.get("age", 35)),
            gender=str(row.get("gender", "male")),
            metro_status=str(row.get("metro_status", "metro")),
            state=state, region=region,
            education=str(row.get("education", "some_college")),
            race=str(row.get("race", "White")),
            political_leaning=str(row.get("political_leaning", "Independent")),
            income_annual=float(row.get("income_annual", 55000)),
            income_percentile=float(row.get("income_percentile", 50)),
            asset_level=str(row.get("asset_level", "medium")),
            debt_has_loan=int(row.get("debt_has_loan", 0)),
            debt_amount=float(row.get("debt_amount", 0)),
            homeownership=str(row.get("homeownership", "renter")),
            has_student_debt=int(row.get("has_student_debt", 0)),
            has_health_insurance=int(row.get("has_health_insurance", 1)),
            employ_status=str(row.get("employ_status", "employed")),
            job_satisfaction=float(row.get("job_satisfaction", 3.0)),
            happiness=float(row.get("happiness", 2.0)),
            life_satisfaction=float(row.get("life_satisfaction", 4.0)),
            future_confidence=float(row.get("future_confidence", 3.0)),
            income_local_rank=float(row.get("income_local_rank", 3.0)),
            social_status=float(row.get("social_status", 3.0)),
            financial_satisfaction=float(row.get("financial_satisfaction", 3.0)),
            work_reward_belief=float(row.get("work_reward_belief", 3.0)),
            mobility_belief=float(row.get("mobility_belief", 3.0)),
            health_status=str(row.get("health_status", "good")),
            social_trust=str(row.get("social_trust", "cautious")),
            is_core=is_core,
        )
