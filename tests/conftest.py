"""Shared pytest fixtures for the ETHOS test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from models.agent import Agent
from models.citizen import Citizen
from models.city import City
from models.street import Street


class MockAIService:
    """Minimal stand-in for AIService that returns deterministic JSON."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {
            "internal_state": {
                "dominant_emotion": "neutral",
                "current_belief_system": "Undecided",
                "psychological_tension": 0.0,
                "happiness": 0.8,
                "integrity": 1.0,
            },
            "reconstructed_logic": "I chose to maintain my routine.",
            "chosen_action": {
                "action_type": "Work",
                "details": "Continues the daily routine.",
            },
            "outbound_action": None,
            "dystopian_decision": {
                "accept_deal": False,
                "abandoned_family_member": "none",
                "utilitarian_justification": "I will not abandon my family.",
            },
            "diary_entry": "A quiet day of honest labour.",
        }

    def generate_creative(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tension: float | None = None,
    ) -> str:
        """Return the configured JSON response as a string."""
        return json.dumps(self.response)


@pytest.fixture
def mock_ai_service() -> MockAIService:
    """Return a default MockAIService instance."""
    return MockAIService()


@pytest.fixture
def sample_citizen() -> Citizen:
    """Return a single, validated Citizen for unit tests."""
    return Citizen(
        id=1,
        name="Test Citizen",
        profession="Farmer",
        wealth=100.0,
        status="married",
        sons=1,
        daughters=1,
    )


@pytest.fixture
def sample_agent(sample_citizen: Citizen, mock_ai_service: MockAIService) -> Agent:
    """Return an Agent wrapping the sample citizen."""
    return Agent(citizen=sample_citizen, ai_service=mock_ai_service)  # type: ignore[arg-type]


@pytest.fixture
def sample_street(sample_agent: Agent) -> Street:
    """Return a Street containing the sample agent as both rep and regular."""
    return Street(
        street_id=1,
        class_label="vulnerable",
        agents=[sample_agent],
        representative=sample_agent,
        regulars=[],
    )


@pytest.fixture
def sample_city(sample_agent: Agent) -> City:
    """Return a City containing the sample agent."""
    city = City(name="Testopolis")
    city.add_agent(sample_agent)
    return city


@pytest.fixture
def temp_json_file(tmp_path: Path) -> Path:
    """Create a temporary JSON file and return its path."""
    path = tmp_path / "test_data.json"
    path.write_text(json.dumps({"key": "value", "number": 42}), encoding="utf-8")
    return path
