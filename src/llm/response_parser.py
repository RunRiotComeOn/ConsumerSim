"""
Parse and validate LLM responses using Pydantic.
"""
import json
import logging
import re
from typing import List

from pydantic import BaseModel, field_validator

from src.agents.base import SurveyResponse

logger = logging.getLogger(__name__)


class LLMSurveyItem(BaseModel):
    PAGO: str
    DUR: str
    PEXP: str
    BUS12: str
    BUS5: str

    @field_validator("PAGO", "PEXP")
    @classmethod
    def validate_better_same_worse(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("better", "same", "worse"):
            raise ValueError(f"Expected better/same/worse, got: {v!r}")
        return v

    @field_validator("DUR", "BUS12", "BUS5")
    @classmethod
    def validate_good_uncertain_bad(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("good", "uncertain", "bad"):
            raise ValueError(f"Expected good/uncertain/bad, got: {v!r}")
        return v

    def to_survey_response(self) -> SurveyResponse:
        return SurveyResponse(
            PAGO=self.PAGO,
            DUR=self.DUR,
            PEXP=self.PEXP,
            BUS12=self.BUS12,
            BUS5=self.BUS5,
        )


def _extract_json(text: str) -> str:
    """Try to extract a JSON array from potentially messy LLM output."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first '[' to last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def parse_llm_response(raw: str, expected_count: int) -> List[SurveyResponse]:
    """
    Parse raw LLM output into a list of SurveyResponse objects.
    Falls back to default responses on parse errors.
    """
    default_response = SurveyResponse()

    try:
        json_str = _extract_json(raw)
        data = json.loads(json_str)

        if not isinstance(data, list):
            raise ValueError("Response is not a JSON array")

        results = []
        for i, item in enumerate(data):
            try:
                validated = LLMSurveyItem(**item)
                results.append(validated.to_survey_response())
            except Exception as e:
                logger.warning(f"Item {i} validation failed: {e}. Using default.")
                results.append(default_response)

        # Pad or truncate to expected_count
        while len(results) < expected_count:
            results.append(default_response)
        return results[:expected_count]

    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}\nRaw: {raw[:200]}")
        return [default_response] * expected_count
