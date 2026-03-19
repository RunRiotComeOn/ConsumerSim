"""
US LLM inference module — uses OpenAI or Deepseek-compatible APIs with US-specific prompts.
"""
import logging
import os
import time
from typing import List

from src.agents.base import SurveyResponse
from src.agents.base_us import USConsumerAgent
from src.llm.prompts_us import SYSTEM_PROMPT_US, build_batch_prompt_us
from src.llm.response_parser import parse_llm_response

logger = logging.getLogger(__name__)


class USLLMInferenceEngine:
    """Handles batch LLM calls for US Core Agents."""

    def __init__(self, config: dict):
        self.config = config["llm"]
        self.batch_size = config["simulation"]["batch_size"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            provider = self.config.get("provider", "openai")
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

            api_key = self.config.get("api_key")
            if not api_key:
                api_key_env = self.config.get("api_key_env", "OPENAI_API_KEY")
                api_key = os.environ.get(api_key_env)
            if not api_key:
                raise EnvironmentError("No LLM API key provided.")

            client_kwargs = {"api_key": api_key}
            base_url = self.config.get("base_url")
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
            if provider == "deepseek":
                logger.info("Using Deepseek-compatible chat endpoint via OpenAI client.")
        return self._client

    def _call_llm(self, user_message: str, n_agents: int) -> List[SurveyResponse]:
        client = self._get_client()
        max_retries = self.config.get("max_retries", 3)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config["model"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_US},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 4096),
                    timeout=self.config.get("timeout", 60),
                )
                raw = response.choices[0].message.content
                return parse_llm_response(raw, n_agents)

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        logger.error("All LLM retries failed. Marking agents as INVALID (will be skipped in Bayesian update).")
        return [SurveyResponse(PAGO="INVALID", DUR="INVALID", PEXP="INVALID",
                               BUS12="INVALID", BUS5="INVALID")] * n_agents

    def run_batch(self, agents: List[USConsumerAgent],
                  macro_context: str = "",
                  prediction_target_month: str = "",
                  survey_cutoff_date: str = "",
                  prior_ics: float = None) -> List[USConsumerAgent]:
        total = len(agents)
        logger.info(f"Running LLM inference on {total} US core agents (batch_size={self.batch_size})...")
        if macro_context:
            logger.info("  (with real-time macro context injected)")

        for i in range(0, total, self.batch_size):
            batch = agents[i : i + self.batch_size]
            user_msg = build_batch_prompt_us(
                batch,
                macro_context=macro_context,
                prediction_target_month=prediction_target_month or None,
                survey_cutoff_date=survey_cutoff_date or None,
                prior_ics=prior_ics,
            )

            logger.debug(f"  Batch {i // self.batch_size + 1}: agents {i}-{i + len(batch) - 1}")
            responses = self._call_llm(user_msg, len(batch))

            for agent, resp in zip(batch, responses):
                agent.response = resp

            if i + self.batch_size < total:
                time.sleep(0.5)

        logger.info("LLM inference complete.")
        return agents
