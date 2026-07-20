"""Initial unit tests for ETHOS simulation engines and utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engines.fear_engine import FearEngine
from engines.heuristic_engine import HeuristicEngine
from models.agent import Agent
from models.street import Street
from utils.helpers import load_json


class TestFearEngine:
    """Tests for the global fear index computation."""

    def test_fear_index_stays_within_bounds(self) -> None:
        """Fear index must always remain within the closed interval [0, 1]."""
        engine = FearEngine()

        # Extreme inputs: heavy vulnerable distress and public shaming.
        feed = [
            "VULNERABLE DISTRESS: Street 1 reported child starvation or bankruptcy today.",
            "PUBLIC SHAMING: [Test - Caught for theft] (Street 1)",
        ]
        active_events = ["drought_rumor", "plague_scare"]

        fear = engine.compute_town_fear_index(
            town_fear_index=0.95,
            town_square_feed=feed,
            active_narrative_events=active_events,
        )

        assert 0.0 <= fear <= 1.0

    def test_fear_index_decays_with_empty_feed(self) -> None:
        """Fear should decay toward zero when there are no reinforcing events."""
        engine = FearEngine()
        fear = engine.compute_town_fear_index(
            town_fear_index=0.5,
            town_square_feed=[],
            active_narrative_events=[],
        )

        assert 0.0 <= fear < 0.5


class TestHeuristicEngine:
    """Tests for deterministic citizen wealth updates."""

    def test_wealth_changes_after_daily_processing(
        self, sample_agent: Agent, sample_street: Street
    ) -> None:
        """Agent wealth must update deterministically after processing one day."""
        engine = HeuristicEngine()
        initial_wealth = sample_agent.wealth

        result = engine.process_agent(
            agent=sample_agent,
            day=1,
            street=sample_street,
            town_fear_index=0.0,
            daily_moral_anomalies=[],
            reputations={},
        )

        assert result["wealth"] == sample_agent.wealth
        assert sample_agent.wealth != initial_wealth or result["income"] == result["cost"]
        assert "income" in result
        assert "cost" in result
        assert "net_delta" in result

    def test_deterministic_event_reproducibility(self, sample_agent: Agent) -> None:
        """The same (agent.id, day) seed must produce the same event outcome."""
        engine = HeuristicEngine()

        event_a = engine._deterministic_event(
            agent=sample_agent, day=5, income=60.0, cost=20.0
        )
        event_b = engine._deterministic_event(
            agent=sample_agent, day=5, income=60.0, cost=20.0
        )

        assert event_a == event_b
        assert "delta" in event_a
        assert "label" in event_a


class TestHelpers:
    """Tests for utility helpers, focusing on JSON parsing."""

    def test_load_json_parses_valid_file(self, temp_json_file: Path) -> None:
        """load_json must correctly parse a valid JSON file."""
        data = load_json(temp_json_file)

        assert data["key"] == "value"
        assert data["number"] == 42

    def test_load_json_rejects_invalid_json(self, tmp_path: Path) -> None:
        """load_json must raise ValueError when the file contains invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(ValueError):
            load_json(bad_file)

    def test_load_json_missing_file_raises(self, tmp_path: Path) -> None:
        """load_json must raise FileNotFoundError for a missing file."""
        missing = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            load_json(missing)
