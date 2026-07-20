"""
Core simulation orchestration engine for ETHOS.

Loads citizens, instantiates agents, wires them into the city's street-based
architecture, and runs the configured number of simulated days.  The
:class:`~models.city.City` controller delegates all business logic to the
specialised engines under :mod:`engines`:

* :class:`~engines.heuristic_engine.HeuristicEngine` for deterministic regular-citizen updates.
* :class:`~engines.representative_engine.RepresentativeEngine` for zoning, telemetry, and representative LLM calls.
* :class:`~engines.fear_engine.FearEngine` for fear index, Town Square feed, and dystopian decrees.

Daily results are persisted to the MySQL database in real time via
:class:`~services.db_repository.DBRepository`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings, get_settings
from models.agent import Agent
from models.citizen import Citizen
from models.city import City
from services.ai_service import AIService
from services.db_repository import DBRepository
from utils.helpers import describe, load_json, save_json
from utils.logger import get_logger

logger = get_logger(__name__)


class SimulationEngine:
    """
    End-to-end controller for a ETHOS simulation run.

    Responsibilities:
      1. Load and validate the citizen registry from JSON.
      2. Build the city, populate it with LLM-backed agents, and organize
         citizens into socio-economic streets with elected representatives.
      3. Run the day loop and persist daily agent metrics and memories.
      4. Emit summary reports.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        ai_service: AIService | None = None,
        db_repository: DBRepository | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ai_service = ai_service or AIService(self.settings)
        self.db = db_repository or DBRepository()
        self.city = City(name="Ethos", ai_service=self.ai_service)
        self.run_history: list[dict] = []

    def load_citizens(self) -> list[Citizen]:
        """Load the raw citizen records from JSON and validate them."""
        raw_data = load_json(self.settings.citizens_path)
        if not isinstance(raw_data, list):
            raise ValueError(
                f"Expected a JSON array of citizens at {self.settings.citizens_path}"
            )

        citizens = [Citizen.model_validate(record) for record in raw_data]
        logger.info("Loaded and validated %d citizens.", len(citizens))
        return citizens

    def build_city(self, citizens: list[Citizen]) -> City:
        """
        Create agents from citizens, register them in the city, and perform
        socio-economic street zoning plus representative selection.
        """
        for citizen in citizens:
            agent = Agent(citizen=citizen, ai_service=self.ai_service)
            self.city.add_agent(agent)

        self.city.organize_streets()

        logger.info(
            "Built %s with population %d across %d streets.",
            self.city.name,
            self.city.population,
            len(self.city.streets),
        )
        return self.city

    def run(self, days: int | None = None) -> dict:
        """
        Execute the full simulation loop.

        Args:
            days: Number of days to simulate; defaults to ``settings.simulation_days``.

        Returns:
            A summary dictionary containing run metadata, street telemetry,
            Town Square history, and individual agent histories.
        """
        days = days if days is not None else self.settings.simulation_days

        try:
            citizens = self.load_citizens()
            self.build_city(citizens)

            logger.info("Starting %d-day simulation.", days)
            for _ in range(days):
                daily_metrics = self.city.simulate_one_day()
                self._capture_daily_memories()
                self._persist_daily_results()
                self.run_history.append(
                    {
                        "day": self.city.current_day,
                        "metrics": daily_metrics,
                        "town_square_feed": list(self.city.town_square_feed),
                        "town_fear_index": self.city.town_fear_index,
                    }
                )

            final_metrics = self.city._aggregate_metrics()
            summary = self._build_summary(days, final_metrics)
            self._persist_results(summary)
            self.city.print_final_report()

            return summary
        finally:
            self.db.close()

    def _capture_daily_memories(self) -> None:
        """
        Trigger memory capture for the current day.

        The hybrid engine normally writes a memory entry for every agent during
        heuristic processing or representative reflection.  This orchestration
        step ensures that any agent whose turn failed still has a fallback entry
        for the day, and it logs the day's capture.
        """
        day = self.city.current_day
        for agent in self.city.agents.values():
            has_entry_for_today = any(
                entry.get("day") == day for entry in agent.memory_stream
            )
            if not has_entry_for_today:
                agent.citizen.add_memory_entry(
                    day=day,
                    event_description=(
                        f"Day {day} passed without a recorded decision."
                    ),
                    agent_reflection=(
                        "You feel unsure about what happened and hope the next day "
                        "brings more clarity for you and your family."
                    ),
                )
        logger.info("Captured daily memories for Day %d.", day)

    def _persist_daily_results(self) -> None:
        """Persist the current day's agent metrics, memories, and interactions."""
        day = self.city.current_day
        daily_metrics: list[dict] = []
        daily_memories: list[dict] = []

        for agent in self.city.agents.values():
            daily_metrics.append(
                {
                    "agent_id": agent.id,
                    "wealth": agent.wealth,
                    "happiness": agent.happiness,
                    "integrity": agent.integrity,
                    "reputation": getattr(agent, "reputation", 1.0),
                    "action_type": getattr(agent, "last_action_type", ""),
                    "fear_index": self.city.town_fear_index,
                }
            )

            for entry in agent.memory_stream:
                if entry.get("day") == day:
                    memory = dict(entry)
                    memory["agent_id"] = agent.id
                    daily_memories.append(memory)

        daily_interactions = [
            interaction
            for interaction in self.city.interaction_history
            if interaction.get("day") == day
        ]

        self.db.add_daily_metrics(day, daily_metrics)
        self.db.add_memory(day, daily_memories)
        self.db.add_agent_interactions(day, daily_interactions)
        logger.info(
            "Persisted daily results for Day %d (%d interactions).",
            day,
            len(daily_interactions),
        )

    def _build_summary(self, days: int, final_metrics: dict) -> dict:
        """Assemble a serialisable run summary."""
        return {
            "run_id": datetime.now(timezone.utc).isoformat(),
            "city": self.city.name,
            "population": self.city.population,
            "days_simulated": days,
            "model": self.ai_service.to_dict(),
            "final_metrics": final_metrics,
            "town_square_feed": self.city.town_square_feed,
            "town_square_history": self.city.town_square_history,
            "public_shaming_log": self.city.public_shaming_log,
            "reputations": self.city.reputations,
            "town_fear_index": self.city.town_fear_index,
            "fear_history": self.city.fear_history,
            "streets": [street.to_dict() for street in self.city.streets],
            "interaction_history": self.city.interaction_history,
            "daily_history": self.run_history,
            # -----------------------------------------------------------------
            # spy_agent.py compatibility: this list must retain the exact fields
            # the CLI inspector expects (id, name, profession, family_size,
            # wealth, happiness, integrity, memory_stream).  It has been extended
            # with the new fluid cognitive fields for full visibility.
            # -----------------------------------------------------------------
            "agent_summaries": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "profession": agent.profession,
                    "family_size": agent.family_size,
                    "wealth": agent.wealth,
                    "happiness": agent.happiness,
                    "integrity": agent.integrity,
                    "reputation": getattr(agent, "reputation", 1.0),
                    "religion": getattr(agent, "religion", "Undecided"),
                    "religion_reason": getattr(agent, "religion_reason", ""),
                    "current_emotion": getattr(agent, "current_emotion", "neutral"),
                    "psychological_tension": getattr(agent, "psychological_tension", 0.0),
                    "last_action_type": getattr(agent, "last_action_type", ""),
                    "last_action_details": getattr(agent, "last_action_details", ""),
                    "last_reconstructed_logic": getattr(agent, "last_reconstructed_logic", ""),
                    "dystopian_decision": getattr(agent, "dystopian_decision", None),
                    "has_accepted_wage_sacrifice_deal": getattr(
                        agent, "has_accepted_wage_sacrifice_deal", False
                    ),
                    "sacrificed_family_member": getattr(
                        agent, "sacrificed_family_member", "none"
                    ),
                    "dystopian_wage_multiplier": getattr(
                        agent, "dystopian_wage_multiplier", 1.0
                    ),
                    "memory_stream": agent.memory_stream,
                }
                for agent in self.city.agents.values()
            ],
        }

    def _persist_results(self, summary: dict) -> None:
        """Write the simulation summary to the data/logs directory."""
        log_dir = Path(self.settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = log_dir / f"simulation_{timestamp}.json"

        save_json(str(output_path), summary)
        logger.info("Simulation results written to %s", output_path)

    @staticmethod
    def describe_metrics(metrics: dict) -> dict:
        """Return descriptive statistics for a metrics dictionary."""
        return describe(metrics)
