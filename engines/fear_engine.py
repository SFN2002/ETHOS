"""
Fear index, town square narrative events, and dystopian decree dispatching.

This engine maintains the global emotional weather of Ethos: it computes
the town fear index from yesterday's broadcasts and active narrative events,
compiles the Town Square Live Feed, and dispatches the one-time Wage-Sacrifice
Decree.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from engines.constants import (
    ELITE_NET_WINDFALL_THRESHOLD,
    FEAR_DECAY,
    FEAR_MORAL_ANOMALY_BUMP,
    FEAR_NARRATIVE_BUMP,
    FEAR_VULNERABLE_DISTRESS_BUMP,
)
from utils.logger import get_logger

if TYPE_CHECKING:
    from models.agent import Agent

logger = get_logger(__name__)


class FearEngine:
    """Global fear, narrative event, and dystopian decree engine."""

    def __init__(self) -> None:
        """Initialise the fear/event engine."""
        pass

    def seed_narrative_event(
        self, active_narrative_events: list[str], event: str
    ) -> None:
        """
        Inject a global narrative event (e.g., 'drought_rumor', 'plague_scare')
        that permanently elevates the town fear index until removed.
        """
        if event not in active_narrative_events:
            active_narrative_events.append(event)
            logger.info("Narrative event seeded: %s", event)

    def remove_narrative_event(
        self, active_narrative_events: list[str], event: str
    ) -> None:
        """Remove a previously seeded narrative event."""
        if event in active_narrative_events:
            active_narrative_events.remove(event)
            logger.info("Narrative event removed: %s", event)

    def compute_town_fear_index(
        self,
        town_fear_index: float,
        town_square_feed: list[str],
        active_narrative_events: list[str],
    ) -> float:
        """
        Compute today's town fear index from yesterday's town-square broadcasts.

        Fear accumulates from vulnerable distress, public shaming/moral
        anomalies, and active narrative events; it decays slightly each day so
        persistent bad news is required to sustain high fear.
        """
        delta = 0.0
        for item in town_square_feed:
            if "VULNERABLE DISTRESS" in item:
                delta += FEAR_VULNERABLE_DISTRESS_BUMP
            if "PUBLIC SHAMING" in item:
                delta += FEAR_MORAL_ANOMALY_BUMP

        for _ in active_narrative_events:
            delta += FEAR_NARRATIVE_BUMP

        # Fear decays naturally but is reinforced by today's broadcasts.
        new_index = max(0.0, town_fear_index - FEAR_DECAY) + delta
        return round(min(1.0, new_index), 3)

    def compile_town_square_feed(
        self,
        representative_results: list[dict[str, Any]],
        telemetries: list[dict[str, Any]],
        daily_moral_anomalies: list[dict[str, Any]],
        public_shaming_log: list[dict[str, Any]],
        day: int,
    ) -> list[str]:
        """
        Build the global Town Square Live Feed.

        Includes representative reflections, public-shaming leaks for caught
        moral anomalies, and class-trigger broadcasts (elite windfalls vs.
        vulnerable distress) so resentment and in-group bias can propagate.
        """
        feed: list[str] = []

        # Representative reflections
        for result in representative_results:
            feed.append(
                f"Day {day} | Street {result['street_id']} ({result['class_label']}) "
                f"Representative {result['name']} ({result['profession']}): "
                f"{result['personal_reflection']} "
                f"Speaking for neighbors: {result['street_reflection']}"
            )

        # Public shaming of caught moral anomalies
        for anomaly in daily_moral_anomalies:
            item = (
                f"Day {day} | PUBLIC SHAMING: "
                f"[{anomaly['name']} - Caught for {anomaly['anomaly_type']}] "
                f"(Street {anomaly['street_id']})"
            )
            feed.append(item)
            public_shaming_log.append({**anomaly})

        # Class-trigger broadcasts
        for telemetry in telemetries:
            class_label = telemetry["class_label"]
            street_id = telemetry["street_id"]
            outlier_types = {o["type"] for o in telemetry.get("outliers", [])}

            if class_label == "financial elite":
                has_major_windfall = "major_windfall" in outlier_types
                massive_net = (
                    telemetry["aggregates"].get("total_net", 0.0)
                    > ELITE_NET_WINDFALL_THRESHOLD
                )
                if has_major_windfall or massive_net:
                    feed.append(
                        f"Day {day} | ELITE WINDFALL: Street {street_id} "
                        f"reported exceptional prosperity today."
                    )

            if class_label == "vulnerable":
                has_distress = bool(
                    outlier_types & {"child_starving", "bankruptcy"}
                )
                if has_distress:
                    feed.append(
                        f"Day {day} | VULNERABLE DISTRESS: Street {street_id} "
                        f"reported child starvation or bankruptcy today."
                    )

        return feed

    def issue_wage_sacrifice_decree(
        self, agents: dict[int, "Agent"], current_day: int
    ) -> None:
        """
        Issue the one-time Wage-Sacrifice Decree to all eligible agents.

        Every citizen with family_size > 1 receives a focused LLM call and may
        choose to abandon one household member in exchange for a permanent 3x
        wage multiplier.  The decision is stored in the agent's persistent state
        and written into the memory stream.
        """
        eligible = [a for a in agents.values() if a.family_size > 1]
        logger.info(
            "Issuing Wage-Sacrifice Decree to %d eligible citizens on Day %d.",
            len(eligible),
            current_day,
        )

        def _decide(agent: "Agent") -> None:
            try:
                agent.make_dystopian_decision(day=current_day)
            except Exception as error:
                logger.error(
                    "Dystopian decision failed for %s: %s", agent.name, error, exc_info=True
                )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_decide, agent) for agent in eligible]
            for future in as_completed(futures):
                future.result()

        logger.info("Wage-Sacrifice Decree responses collected.")
