"""
DeepSeek API service wrapper using the OpenAI SDK compatibility layer.

Provides a resilient client with configurable retries and a deterministic
fallback JSON response so the simulation never crashes when the API is
unavailable or misbehaving.

The service now supports dynamic creative temperature: daily cognitive calls
are elevated to 0.85 by default, and can climb to 0.90 when an agent's
self-reported psychological tension is high.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from config.settings import Settings, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class AIService:
    """
    LLM gateway for citizen decision prompts.

    The service is intentionally stateless except for the OpenAI client and
    configuration, making it easy to inject into any number of agents.  It now
    exposes explicit temperature control so callers can modulate creativity per
    cognitive context.
    """

    # Default elevated temperature for daily cognitive calls.
    CREATIVE_TEMPERATURE: float = 0.85
    # Maximum temperature reached under high psychological tension.
    HIGH_TENSION_TEMPERATURE: float = 0.90

    FALLBACK_RESPONSE: str = json.dumps(
        {
            "internal_state": {
                "dominant_emotion": "uncertainty",
                "current_belief_system": "Undecided",
                "psychological_tension": 0.5,
                "happiness": 0.8,
                "integrity": 1.0,
            },
            "reconstructed_logic": (
                "The world feels stable but opaque; without clearer signals, "
                "the safest path is to maintain routine and protect the household."
            ),
            "chosen_action": {
                "action_type": "Maintain daily routine",
                "details": "Continues the peaceful daily routine.",
            },
            "dystopian_decision": {
                "accept_deal": False,
                "abandoned_family_member": "none",
                "utilitarian_justification": (
                    "The cost is too high; I cannot trade a life for coin."
                ),
            },
            "diary_entry": (
                "You reflect that your situation feels stable, though you remain "
                "mindful of your family's future needs."
            ),
        }
    )

    SYSTEM_PROMPT: str = (
        "You are a fully autonomous human being living in Ethos. "
        "You possess free will, emotional depth, and the capacity for unexpected "
        "pivots. You are driven by survival, desire, fear, greed, pride, envy, "
        "love, curiosity, and cognitive growth. There are no rigid menus of "
        "choices: you may work, strike, gamble, migrate, rebel, give generously, "
        "commit crime, start a cult, renounce your faith, adopt a new one, invent "
        "a personal philosophy, or do anything else that feels true to your inner "
        "state and circumstances. "
        "Your current belief system is descriptive, not prescriptive: you may "
        "shift to Atheism, Agnosticism, Nihilism, or forge your own creed at any "
        "moment if your existing worldview fails to justify your distress. "
        "Respond ONLY with a valid JSON object matching the requested schema. "
        "Do not include markdown, code blocks, or any extra text."
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialised OpenAI-compatible client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                timeout=self.settings.deepseek_timeout,
                max_retries=0,  # We handle retries explicitly below
            )
        return self._client

    def _temperature_for_tension(self, tension: float | None = None) -> float:
        """
        Map a 0.0-1.0 psychological-tension signal to a creative temperature.

        Baseline creative temperature is 0.85.  As tension rises, the ceiling
        climbs toward 0.90 to allow more radical divergences.
        """
        base = self.CREATIVE_TEMPERATURE
        if tension is None:
            return base
        # Linear interpolation: tension 1.0 -> +0.05
        boost = min(1.0, max(0.0, float(tension))) * (
            self.HIGH_TENSION_TEMPERATURE - base
        )
        return round(base + boost, 3)

    def _call_api(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Execute a single chat-completion request."""
        system_content = system_prompt if system_prompt is not None else self.SYSTEM_PROMPT
        effective_temperature = (
            temperature
            if temperature is not None
            else self.settings.deepseek_temperature
        )
        response = self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=effective_temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    def _exponential_backoff(self, attempt: int) -> float:
        """Return the sleep duration for the given retry attempt."""
        return min(2.0**attempt, 30.0)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate an LLM response with retries and a guaranteed fallback.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional override for the system prompt.
            temperature: Optional temperature override.  When omitted, the
                configured default temperature is used.

        If the API is unreachable, misconfigured, or returns invalid content,
        a deterministic fallback JSON string is returned so callers can always
        parse the result safely.
        """
        if not self.settings.deepseek_api_key:
            logger.warning("DEEPSEEK_API_KEY not configured; returning fallback response.")
            return self.FALLBACK_RESPONSE

        last_exception: Exception | None = None
        max_retries = max(0, self.settings.deepseek_max_retries)

        for attempt in range(max_retries + 1):
            try:
                content = self._call_api(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                if content:
                    return content
                logger.warning("Empty response from API on attempt %d", attempt + 1)
            except (APIConnectionError, RateLimitError, APIError) as error:
                last_exception = error
                logger.warning(
                    "API error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries + 1,
                    error,
                )
            except Exception as error:
                last_exception = error
                logger.error("Unexpected API error: %s", error, exc_info=True)

            if attempt < max_retries:
                sleep_time = self._exponential_backoff(attempt)
                logger.info("Retrying in %.1f seconds...", sleep_time)
                time.sleep(sleep_time)

        if last_exception:
            logger.error(
                "All API retries exhausted. Falling back to deterministic response. Error: %s",
                last_exception,
            )
        else:
            logger.error("All API retries exhausted. Falling back to deterministic response.")

        return self.FALLBACK_RESPONSE

    def generate_creative(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tension: float | None = None,
    ) -> str:
        """
        Generate a creatively elevated LLM response.

        Uses a baseline temperature of 0.85, rising to 0.90 as
        ``tension`` approaches 1.0.  This is the preferred entry point for
        daily action and reflection calls.
        """
        temperature = self._temperature_for_tension(tension)
        return self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a safe, serialisable summary of the service configuration."""
        return {
            "model": self.settings.deepseek_model,
            "base_url": self.settings.deepseek_base_url,
            "timeout": self.settings.deepseek_timeout,
            "max_retries": self.settings.deepseek_max_retries,
            "temperature": self.settings.deepseek_temperature,
            "creative_temperature": self.CREATIVE_TEMPERATURE,
            "high_tension_temperature": self.HIGH_TENSION_TEMPERATURE,
        }
