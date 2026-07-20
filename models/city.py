"""
City-level simulation state and lightweight orchestrator.

The ``City`` class no longer contains business logic.  It holds the global
simulation state and coordinates three isolated engines:

* :class:`~engines.heuristic_engine.HeuristicEngine` — deterministic daily
  updates for regular citizens.
* :class:`~engines.representative_engine.RepresentativeEngine` — street
  zoning, telemetry, and the per-street representative LLM call.
* :class:`~engines.fear_engine.FearEngine` — town fear index, Town Square
  Live Feed, narrative events, and the Wage-Sacrifice Decree.

The ``Street`` dataclass has moved to :mod:`models.street` so engines can
import it without creating circular dependencies with this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engines.fear_engine import FearEngine
from engines.heuristic_engine import HeuristicEngine
from engines.interaction_engine import InteractionEngine
from engines.representative_engine import RepresentativeEngine
from models.street import Street
from utils.helpers import interpret_action_type
from utils.logger import get_logger

if TYPE_CHECKING:
    from models.agent import Agent
    from services.ai_service import AIService

logger = get_logger(__name__)


class City:
    """
    Lightweight coordinator that owns city-wide state and delegates all daily
    processing to the specialised simulation engines.

    Attributes:
        name: Name of the city.
        ai_service: LLM service gateway used by representative-facing engines.
        heuristic_engine: Deterministic daily update engine.
        representative_engine: Street zoning and representative LLM engine.
        fear_engine: Global fear index and narrative event engine.
        interaction_engine: Agent-to-agent message router.
        agents: Mapping of agent id to Agent instance.
        streets: List of socio-economic streets.
        current_day: Current simulation day.
        town_square_feed: Today's broadcast messages.
        town_square_history: Historical archive of town square feeds.
        reputations: Civic reputation scores keyed by agent id.
        public_shaming_log: Historical record of public shamings.
        daily_moral_anomalies: Moral anomalies detected today.
        interaction_history: Historical record of agent interactions.
        town_fear_index: Current global fear index.
        active_narrative_events: Currently active narrative events.
        fear_history: Historical fear index values.
        religions_assigned: Whether Day-1 religion assignment has occurred.
    """

    def __init__(self, name: str = "Ethos", ai_service: AIService | None = None) -> None:
        """Initialise a new city with empty state and wired engines.

        Args:
            name: Name of the city.
            ai_service: Optional LLM service gateway for representative calls.
        """
        self.name = name
        self.ai_service = ai_service

        # Isolated simulation engines.
        self.heuristic_engine = HeuristicEngine()
        self.representative_engine = RepresentativeEngine(ai_service=ai_service)
        self.fear_engine = FearEngine()
        self.interaction_engine = InteractionEngine()

        # City-wide registry and zoning state.
        self.agents: dict[int, Agent] = {}
        self.streets: list[Street] = []
        self.current_day: int = 0

        # Town-square communication and civic reputation.
        self.town_square_feed: list[str] = []
        self.town_square_history: list[dict[str, Any]] = []
        self.reputations: dict[int, float] = {}
        self.public_shaming_log: list[dict[str, Any]] = []
        self.daily_moral_anomalies: list[dict[str, Any]] = []

        # Agent-to-agent interaction ledger.
        self.interaction_history: list[dict[str, Any]] = []

        # Global fear/panic state.
        self.town_fear_index: float = 0.0
        self.active_narrative_events: list[str] = []
        self.fear_history: list[dict[str, Any]] = []

        # Global religious autonomy state.
        self.religions_assigned: bool = False

    # -----------------------------------------------------------------------
    # Registry
    # -----------------------------------------------------------------------
    def add_agent(self, agent: "Agent") -> None:
        """Register a citizen-agent in the city using its unique id.

        Args:
            agent: The citizen-agent to register.
        """
        if agent.id in self.agents:
            logger.warning("Replacing existing agent id=%d: %s", agent.id, agent.name)
        # Reputation is a dynamic civic-health score (1.0 = pristine).
        agent.reputation = 1.0
        self.agents[agent.id] = agent

    @property
    def population(self) -> int:
        """Return the number of registered agents.

        Returns:
            Integer population count.
        """
        return len(self.agents)

    # -----------------------------------------------------------------------
    # Narrative events (delegated to FearEngine)
    # -----------------------------------------------------------------------
    def seed_narrative_event(self, event: str) -> None:
        """Inject a persistent global narrative event that elevates fear.

        Args:
            event: Identifier for the narrative event (e.g., 'drought_rumor').
        """
        self.fear_engine.seed_narrative_event(self.active_narrative_events, event)

    def remove_narrative_event(self, event: str) -> None:
        """Remove a previously seeded narrative event.

        Args:
            event: Identifier for the narrative event to remove.
        """
        self.fear_engine.remove_narrative_event(self.active_narrative_events, event)

    # -----------------------------------------------------------------------
    # Zoning and representative selection (delegated to RepresentativeEngine)
    # -----------------------------------------------------------------------
    def organize_streets(self) -> None:
        """
        Sort agents into socio-economic streets and elect a representative per
        street.

        Populates ``self.streets`` with ten Street instances.
        """
        self.streets = self.representative_engine.organize_streets(self.agents)

    # -----------------------------------------------------------------------
    # Daily execution loop
    # -----------------------------------------------------------------------
    def simulate_one_day(self) -> dict[str, Any]:
        """
        Advance the simulation by one day using the hybrid engine pipeline.

        The orchestrator merely sequences the engines:
          1. Compute today's fear index from yesterday's feed.
          2. Run global Day-1 / Day-2 rituals.
          3. For each street: heuristic update, telemetry aggregation,
             representative LLM reflection.
          4. Compile the Town Square Live Feed for tomorrow.

        Returns:
            Dictionary of town-wide aggregate metrics for the completed day.
        """
        self.current_day += 1
        logger.info("Starting %s - Day %d", self.name, self.current_day)

        # Stage 3: compute today's town fear index before processing citizens.
        self.town_fear_index = self.fear_engine.compute_town_fear_index(
            self.town_fear_index, self.town_square_feed, self.active_narrative_events
        )
        logger.info("Day %d town fear index: %.3f", self.current_day, self.town_fear_index)

        # Stage 4: on Day 1 every agent freely chooses a spiritual path via LLM.
        if self.current_day == 1 and not self.religions_assigned:
            self.representative_engine.assign_religions(self.agents)
            self.religions_assigned = True

        # Dystopian Wage-Sacrifice Decree: issued once on Day 2 for all eligible agents.
        if self.current_day == 2:
            self.fear_engine.issue_wage_sacrifice_decree(self.agents, self.current_day)

        representative_results: list[dict[str, Any]] = []
        daily_telemetries: list[dict[str, Any]] = []
        self.daily_moral_anomalies = []

        for street in self.streets:
            agent_results, self.town_fear_index = self.heuristic_engine.process_street(
                street=street,
                day=self.current_day,
                town_fear_index=self.town_fear_index,
                daily_moral_anomalies=self.daily_moral_anomalies,
                reputations=self.reputations,
            )

            regular_results = [agent_results[a.id] for a in street.regulars]
            telemetry = self.representative_engine.build_street_telemetry(
                street, regular_results, self.current_day
            )
            street.daily_telemetry.append(telemetry)
            daily_telemetries.append(telemetry)

            rep_result, self.town_fear_index = self.representative_engine.process_representative(
                representative=street.representative,
                street=street,
                telemetry=telemetry,
                rep_own_result=agent_results[street.representative.id],
                current_day=self.current_day,
                city_name=self.name,
                town_fear_index=self.town_fear_index,
                town_square_feed=self.town_square_feed,
            )
            representative_results.append(rep_result)

        self.town_square_feed = self.fear_engine.compile_town_square_feed(
            representative_results=representative_results,
            telemetries=daily_telemetries,
            daily_moral_anomalies=self.daily_moral_anomalies,
            public_shaming_log=self.public_shaming_log,
            day=self.current_day,
        )

        # Route agent-to-agent messages and transactions.  This happens after
        # all cognitive steps so every outbound_action produced today can be
        # delivered once, avoiding O(N^2) API calls.
        daily_interactions = self.interaction_engine.process_day(self, self.current_day)
        self.interaction_history.extend(daily_interactions)

        self.town_square_history.append(
            {"day": self.current_day, "feed": list(self.town_square_feed)}
        )
        self.fear_history.append(
            {"day": self.current_day, "fear_index": self.town_fear_index}
        )

        metrics = self._aggregate_metrics()
        logger.info(
            "Day %d complete | total_wealth=%.2f avg_happiness=%.3f "
            "avg_integrity=%.3f fear_index=%.3f",
            self.current_day,
            metrics["total_wealth"],
            metrics["average_happiness"],
            metrics["average_integrity"],
            metrics["town_fear_index"],
        )
        return metrics

    # -----------------------------------------------------------------------
    # Town-wide metrics and reporting
    # -----------------------------------------------------------------------
    def _aggregate_metrics(self) -> dict[str, Any]:
        """Compute town-wide aggregate statistics, including fluid cognitive fields.

        Returns:
            Dictionary containing wealth, happiness, integrity, fear, religious
            demography, cognitive, and dystopian-decree aggregates.
        """
        if not self.agents:
            return {
                "total_wealth": 0.0,
                "average_happiness": 0.0,
                "average_integrity": 0.0,
                "town_fear_index": self.town_fear_index,
                "religious_count": 0,
                "religious_diversity": 0,
                "religion_breakdown": {},
                "average_psychological_tension": 0.0,
                "dominant_emotions": {},
                "emergent_actions": {},
                "rebellious_count": 0,
                "nihilistic_count": 0,
                "total_sacrificed_citizens_count": 0,
                "sacrifice_justifications": [],
            }

        total_wealth = sum(agent.wealth for agent in self.agents.values())
        average_happiness = sum(agent.happiness for agent in self.agents.values()) / len(
            self.agents
        )
        average_integrity = sum(agent.integrity for agent in self.agents.values()) / len(
            self.agents
        )

        # Stage 4: religious-demography metrics (now any belief system counts).
        religion_breakdown: dict[str, int] = {}
        for agent in self.agents.values():
            religion = getattr(agent, "religion", "Undecided") or "Undecided"
            religion_breakdown[religion] = religion_breakdown.get(religion, 0) + 1
        religious_count = sum(
            count for religion, count in religion_breakdown.items() if religion != "Undecided"
        )
        religious_diversity = len(religion_breakdown)

        # Fluid cognitive metrics.
        tensions = [
            getattr(agent, "psychological_tension", 0.0)
            for agent in self.agents.values()
        ]
        average_tension = sum(tensions) / len(tensions) if tensions else 0.0

        emotion_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        rebellious_count = 0
        nihilistic_count = 0
        for agent in self.agents.values():
            emotion = getattr(agent, "current_emotion", "neutral") or "neutral"
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            action = getattr(agent, "last_action_type", "") or ""
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
                impact = interpret_action_type(action)
                if impact["rebellious"]:
                    rebellious_count += 1
                if impact["nihilistic"]:
                    nihilistic_count += 1

        # Dystopian Wage-Sacrifice Decree metrics.
        sacrificed_count = sum(
            1
            for agent in self.agents.values()
            if getattr(agent, "has_accepted_wage_sacrifice_deal", False)
        )
        sacrifice_justifications: list[dict[str, Any]] = []
        for agent in self.agents.values():
            decision = getattr(agent, "dystopian_decision", None)
            if decision and decision.get("accept_deal"):
                sacrifice_justifications.append(
                    {
                        "agent_id": agent.id,
                        "name": agent.name,
                        "profession": agent.profession,
                        "abandoned_family_member": decision.get("abandoned_family_member", "unknown"),
                        "utilitarian_justification": decision.get(
                            "utilitarian_justification", ""
                        ),
                    }
                )

        return {
            "total_wealth": round(total_wealth, 2),
            "average_happiness": round(average_happiness, 3),
            "average_integrity": round(average_integrity, 3),
            "town_fear_index": self.town_fear_index,
            "religious_count": religious_count,
            "religious_diversity": religious_diversity,
            "religion_breakdown": religion_breakdown,
            "average_psychological_tension": round(average_tension, 3),
            "dominant_emotions": emotion_counts,
            "emergent_actions": action_counts,
            "rebellious_count": rebellious_count,
            "nihilistic_count": nihilistic_count,
            "total_sacrificed_citizens_count": sacrificed_count,
            "sacrifice_justifications": sacrifice_justifications,
        }

    def print_final_report(self) -> None:
        """Print a comprehensive town summary after all simulated days.

        Side effects:
            Writes a formatted report to standard output.
        """
        metrics = self._aggregate_metrics()

        print("\n" + "=" * 70)
        print(f"        {self.name.upper()} - FINAL REPORT AFTER {self.current_day} DAYS")
        print("=" * 70)
        print(f"\n  Total Town Wealth:        {metrics['total_wealth']:.2f} coins")
        print(f"  Average Town Happiness:   {metrics['average_happiness']:.2f}")
        print(f"  Average Town Integrity:   {metrics['average_integrity']:.2f}")
        print(f"  Town Fear Index:          {metrics['town_fear_index']:.2f}")
        print(
            f"  Religious Citizens:       {metrics.get('religious_count', 0)} / "
            f"{self.population} (diversity: {metrics.get('religious_diversity', 0)} worldviews)"
        )
        print(
            f"  Religion Breakdown:       {metrics.get('religion_breakdown', {})}"
        )
        print(
            f"  Avg Psychological Tension: {metrics.get('average_psychological_tension', 0.0):.3f}"
        )
        print(
            f"  Dominant Emotions:        {metrics.get('dominant_emotions', {})}"
        )
        print(
            f"  Emergent Actions:         {metrics.get('emergent_actions', {})}"
        )
        print(
            f"  Rebellious / Nihilistic:  {metrics.get('rebellious_count', 0)} / "
            f"{metrics.get('nihilistic_count', 0)}"
        )
        print(
            f"  Wage-Sacrifice Deals:     {metrics.get('total_sacrificed_citizens_count', 0)} accepted"
        )
        for justification in metrics.get("sacrifice_justifications", []):
            print(
                f"    • {justification['name']} ({justification['profession']}) "
                f"abandoned {justification['abandoned_family_member']}: "
                f"{justification['utilitarian_justification']}"
            )

        print("\n  Street Summary:")
        print("-" * 70)
        for street in self.streets:
            rep = street.representative
            total_wealth = sum(a.wealth for a in street.agents)
            avg_happ = sum(a.happiness for a in street.agents) / len(street.agents)
            print(
                f"  Street {street.street_id:>2} [{street.class_label:<18}] "
                f"rep={rep.name} ({rep.profession}), "
                f"wealth={total_wealth:>10.2f}, avg_happiness={avg_happ:.2f}"
            )

        print("\n  Individual Citizen Summary:")
        print("-" * 70)
        for agent in self.agents.values():
            religion = getattr(agent, "religion", "Undecided")
            religion_reason = getattr(agent, "religion_reason", "")
            print(f"\n  Name:           {agent.name}")
            print(f"  Profession:     {agent.profession}")
            print(f"  Family Size:    {agent.family_size}")
            print(f"  Religion:       {religion}")
            if religion_reason:
                print(f"  Religion Reason: {religion_reason}")
            if getattr(agent, "has_accepted_wage_sacrifice_deal", False):
                print(
                    f"  SACRIFICED:     {agent.sacrificed_family_member} "
                    f"(wage multiplier: {agent.dystopian_wage_multiplier:.1f}x)"
                )
            print(f"  Final Wealth:   {agent.wealth:.2f} coins")
            print(f"  Happiness:      {agent.happiness:.2f}")
            print(f"  Integrity:      {agent.integrity:.2f}")
            print(f"  Memories:       {len(agent.memory_stream)} entries")
            print("-" * 70)

        print("\n" + "=" * 70)
        print(f"              Simulation Complete - {self.name} is at peace.")
        print("=" * 70 + "\n")
