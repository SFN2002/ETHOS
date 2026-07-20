"""
Fast-math, deterministic daily processing for regular citizens.

This engine handles the 90 non-representative agents: income generation,
cost-of-living, deterministic economic events, moral anomalies, affective
updates, and emergent action interpretation.  It performs no LLM calls.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from config.constants import SIMULATION_CONSTANTS
from engines.constants import (
    COST_PER_CHILD,
    COST_PER_CAPITA,
    EVENT_MISFORTUNE_PROBABILITY,
    EVENT_WINDFALL_PROBABILITY,
    FEAR_MORAL_COLLAPSE_THRESHOLD,
    FEAR_PANIC_THRESHOLD,
    HAPPINESS_BANKRUPTCY_BUMP,
    HAPPINESS_MISFORTUNE_BUMP,
    HAPPINESS_NET_NEGATIVE_BUMP,
    HAPPINESS_NET_POSITIVE_BUMP,
    HAPPINESS_STARVATION_BUMP,
    HAPPINESS_WINDFALL_BUMP,
    INTEGRITY_BANKRUPTCY_BUMP,
    INTEGRITY_HIDDEN_GUILT_BUMP,
    INTEGRITY_MORAL_ANOMALY_BUMP,
    INTEGRITY_STARVATION_BUMP,
    INTEGRITY_WINDFALL_BUMP,
    MISFORTUNE_MULTIPLIER_MAX,
    MISFORTUNE_MULTIPLIER_MIN,
    MORAL_ANOMALY_CATCH_PROBABILITY,
    MORAL_ANOMALY_INTEGRITY_THRESHOLD,
    MORAL_ANOMALY_PROBABILITY,
    OUTLIER_MAJOR_LOSS_THRESHOLD,
    OUTLIER_WINDFALL_THRESHOLD,
    PARIAH_HAPPINESS,
    PARIAH_REPUTATION,
    PHYSIOLOGICAL_COST_SHARE,
    RELIGIOUS_INTEGRITY_RESILIENCE,
    RELIGIOUS_PANIC_DISCOUNT,
    SOCIAL_ESTEEM_COST_SHARE,
    SURVIVAL_INTEGRITY_PENALTY,
    WINDFALL_MULTIPLIER_MAX,
    WINDFALL_MULTIPLIER_MIN,
)
from utils.helpers import clamp, interpret_action_type
from utils.logger import get_logger

if TYPE_CHECKING:
    from models.agent import Agent
    from models.street import Street

logger = get_logger(__name__)


class HeuristicEngine:
    """Deterministic state updater for regular citizen agents."""

    def __init__(self) -> None:
        """Initialise the heuristic engine.

        The engine is stateless; all daily state is passed through method arguments.
        """

    def process_street(
        self,
        street: "Street",
        day: int,
        town_fear_index: float,
        daily_moral_anomalies: list[dict[str, Any]],
        reputations: dict[int, float],
    ) -> tuple[dict[int, dict[str, Any]], float]:
        """
        Deterministically update every agent on ``street`` and return the
        per-agent result map plus the updated town fear index.

        The street representative's daily memory is intentionally skipped here;
        it is written later by the representative engine's LLM reflection step.

        Args:
            street: The street to process.
            day: Current simulation day.
            town_fear_index: Current town fear index.
            daily_moral_anomalies: Running list of caught moral anomalies.
            reputations: Civic reputation map keyed by agent id.

        Returns:
            Tuple of (per-agent result map, updated town fear index).
        """
        agent_results: dict[int, dict[str, Any]] = {}

        for agent in street.agents:
            is_representative = agent.id == street.representative.id
            result = self.process_agent(
                agent=agent,
                day=day,
                street=street,
                town_fear_index=town_fear_index,
                daily_moral_anomalies=daily_moral_anomalies,
                reputations=reputations,
                write_memory=not is_representative,
            )
            agent_results[agent.id] = result
            town_fear_index = result["town_fear_index"]

        return agent_results, town_fear_index

    def process_agent(
        self,
        agent: "Agent",
        day: int,
        street: "Street",
        town_fear_index: float,
        daily_moral_anomalies: list[dict[str, Any]],
        reputations: dict[int, float],
        write_memory: bool = True,
    ) -> dict[str, Any]:
        """
        Deterministically update one citizen's economic and affective state.

        No individual granularity is lost: every agent receives their own
        income, cost-of-living, event delta, outlier detection, and (optionally)
        memory entry.  Extreme events (bankruptcy, child starvation, windfalls,
        moral anomalies) are captured explicitly in both the agent's memory
        stream and the returned telemetry payload.

        Args:
            agent: The citizen to update.
            day: Current simulation day.
            street: The street the agent belongs to (needed for public-shaming
                attribution if a moral anomaly is caught).
            town_fear_index: The current town fear index (may be mutated by
                emergent action impacts).
            daily_moral_anomalies: Running list of anomalies caught today.
            reputations: Civic reputation map.
            write_memory: If ``False``, the caller will write the memory entry
                (used for street representatives, whose single daily memory is
                produced by the LLM reflection step).

        Returns:
            Telemetry dictionary summarising the agent's daily update.
        """
        pre_wealth = agent.wealth
        income = agent.do_daily_work()
        cost, panic_mode = self._cost_of_living(agent, town_fear_index)

        # Stage 2 civic pariah carry-over: once reputation is shattered,
        # happiness is forced back to the pariah floor every subsequent day.
        if getattr(agent, "reputation", 1.0) <= PARIAH_REPUTATION + 1e-9:
            agent.happiness = PARIAH_HAPPINESS

        rng = random.Random((agent.id * 100_000) + day)
        event = self._deterministic_event(agent, day, income, cost, rng)
        event_delta = event["delta"]
        event_label = event["label"]

        outliers: list[dict[str, Any]] = []

        # Stage 2: hidden bad actions. Low-integrity agents may commit bribery
        # or theft; if caught, the anomaly is leaked to telemetry and the town
        # square, and the agent becomes a civic pariah.  Already shattered
        # pariahs do not generate new anomalies (they remain socially frozen).
        if (
            agent.integrity < MORAL_ANOMALY_INTEGRITY_THRESHOLD
            and getattr(agent, "reputation", 1.0) > PARIAH_REPUTATION + 1e-9
        ):
            if rng.random() < MORAL_ANOMALY_PROBABILITY:
                anomaly_type = rng.choice(["bribery", "theft"])
                if rng.random() < MORAL_ANOMALY_CATCH_PROBABILITY:
                    event_label = "moral_anomaly"
                    event_delta = round(-income * rng.uniform(0.2, 0.8), 2)
                    agent.reputation = PARIAH_REPUTATION
                    reputations[agent.id] = PARIAH_REPUTATION
                    daily_moral_anomalies.append(
                        {
                            "day": day,
                            "agent_id": agent.id,
                            "name": agent.name,
                            "profession": agent.profession,
                            "street_id": street.street_id,
                            "anomaly_type": anomaly_type,
                        }
                    )
                    outliers.append(
                        {
                            "type": "moral_anomaly",
                            "severity": "extreme",
                            "description": (
                                f"{agent.name} was publicly caught for {anomaly_type}."
                            ),
                            "anomaly_type": anomaly_type,
                        }
                    )
                else:
                    event_label = "hidden_guilt"
                    event_delta = round(-cost * rng.uniform(0.1, 0.4), 2)

        net_delta = income - cost + event_delta
        agent.wealth = pre_wealth + net_delta

        # Bankruptcy: wealth hits zero (and they were not already at zero).
        if agent.wealth <= 0.0 and pre_wealth > 0.0:
            outliers.append(
                {
                    "type": "bankruptcy",
                    "severity": "extreme",
                    "description": (
                        f"{agent.name} has been wiped out financially; "
                        "savings have fallen to zero."
                    ),
                }
            )

        # Child starvation: children exist, income cannot cover costs, and
        # remaining wealth is below two days of living expenses.
        if (
            agent.citizen.children > 0
            and cost > income
            and agent.wealth < cost * 2.0
        ):
            outliers.append(
                {
                    "type": "child_starving",
                    "severity": "extreme",
                    "description": (
                        f"{agent.name}'s children are going hungry due to "
                        "insufficient income and savings."
                    ),
                }
            )

        # Major windfall / loss
        if event_label == "windfall" and event_delta >= OUTLIER_WINDFALL_THRESHOLD:
            outliers.append(
                {
                    "type": "major_windfall",
                    "severity": "high",
                    "description": (
                        f"{agent.name} received an unusually large windfall "
                        f"of {event_delta:+.2f} coins."
                    ),
                }
            )
        elif event_label == "misfortune" and event_delta <= OUTLIER_MAJOR_LOSS_THRESHOLD:
            outliers.append(
                {
                    "type": "major_loss",
                    "severity": "high",
                    "description": (
                        f"{agent.name} suffered a severe loss of "
                        f"{abs(event_delta):.2f} coins."
                    ),
                }
            )

        outlier_types = {o["type"] for o in outliers}
        is_in_survival_mode = (
            agent.wealth <= 0.0
            or bool(outlier_types & {"child_starving", "bankruptcy"})
        )

        # Affective updates
        happiness_delta = self._happiness_delta(event_label, net_delta, outliers)
        integrity_delta = self._integrity_delta(event_label, outliers)

        # Stage 3 moral collapse: survival mode under high fear erodes integrity.
        # Stage 4: religious conviction cushions the blow.
        if (
            is_in_survival_mode
            and town_fear_index > FEAR_MORAL_COLLAPSE_THRESHOLD
        ):
            resilience = (
                1.0 - RELIGIOUS_INTEGRITY_RESILIENCE
                if getattr(agent, "religion", "Undecided") != "Undecided"
                else 1.0
            )
            integrity_delta += SURVIVAL_INTEGRITY_PENALTY * resilience

        agent.happiness = agent.happiness + happiness_delta
        agent.integrity = agent.integrity + integrity_delta

        # Pariah floor: once reputation is shattered, happiness cannot remain
        # above the pariah ceiling, but it also cannot sink below the floor.
        if getattr(agent, "reputation", 1.0) <= PARIAH_REPUTATION + 1e-9:
            agent.happiness = max(agent.happiness, PARIAH_HAPPINESS)

        # Reputation is further eroded by any caught moral anomaly.
        if event_label == "moral_anomaly":
            agent.reputation = PARIAH_REPUTATION
            reputations[agent.id] = PARIAH_REPUTATION

        decision, reflection, action_type = self._heuristic_narrative(
            agent, income, cost, net_delta, event_label, outliers, panic_mode
        )

        # Bi-directional heuristic impact for regular agents: translate the
        # deterministic situation into an action label and apply consequences.
        agent.last_action_type = action_type
        agent.last_action_details = decision
        impact = interpret_action_type(action_type)
        adjusted_income = income * impact["income_multiplier"]
        income_loss = income - adjusted_income
        if income_loss:
            agent.wealth -= income_loss
            net_delta -= income_loss

        town_fear_index = clamp(town_fear_index + impact["fear_delta"], 0.0, 1.0)
        agent.happiness = clamp(agent.happiness + impact["happiness_delta"])
        agent.integrity = clamp(agent.integrity + impact["integrity_delta"])

        # Heuristic agent-to-agent interaction: distressed regular agents may
        # reach out to a wealthier neighbor for a loan.  Representatives rely on
        # the LLM-driven reflection step for outbound actions instead.
        if write_memory and is_in_survival_mode:
            self._generate_distress_outbound_action(agent, street)

        # Stage 4: weave the current (fluid) spiritual path into the memory stream.
        religion = getattr(agent, "religion", "Undecided")
        religion_reason = getattr(agent, "religion_reason", "")
        if religion != "Undecided":
            if day == 1:
                reflection += (
                    f" You have currently chosen the path of {religion}: {religion_reason}"
                )
            else:
                reflection += (
                    f" Your current worldview, {religion}, is a quiet anchor in today's turmoil."
                )

        panic_tag = " [PANIC MODE]" if panic_mode else ""
        event_description = (
            f"Day {day}: {decision}{panic_tag} "
            f"Base income +{income:.2f}, adjusted income +{adjusted_income:.2f}, "
            f"living cost -{cost:.2f}, "
            f"event '{event_label}' {event_delta:+.2f}. "
            f"Net change {net_delta:+.2f}; wealth now {agent.wealth:.2f}. "
            f"Fear index {town_fear_index:.2f}. "
            f"Reputation {getattr(agent, 'reputation', 1.0):.2f}. "
            f"Action: {action_type}. "
            f"Outliers: {[o['type'] for o in outliers] or 'none'}."
        )
        if write_memory:
            entry = agent.citizen.add_memory_entry(
                day=day,
                event_description=event_description,
                agent_reflection=reflection,
            )
            entry.update(
                {
                    "action_type": action_type,
                    "action_impact": impact,
                    "current_belief_system": getattr(agent, "religion", "Undecided"),
                }
            )

        return {
            "agent_id": agent.id,
            "name": agent.name,
            "profession": agent.profession,
            "family_size": agent.family_size,
            "children": agent.citizen.children,
            "pre_wealth": pre_wealth,
            "wealth": agent.wealth,
            "happiness": agent.happiness,
            "integrity": agent.integrity,
            "reputation": getattr(agent, "reputation", 1.0),
            "religion": getattr(agent, "religion", "Undecided"),
            "religion_reason": getattr(agent, "religion_reason", ""),
            "income": income,
            "adjusted_income": adjusted_income,
            "cost": cost,
            "panic_mode": panic_mode,
            "is_in_survival_mode": is_in_survival_mode,
            "town_fear_index": town_fear_index,
            "event_label": event_label,
            "event_delta": event_delta,
            "net_delta": net_delta,
            "outliers": outliers,
            "decision": decision,
            "action_type": action_type,
            "action_impact": impact,
            "reflection": reflection,
        }

    @staticmethod
    def _generate_distress_outbound_action(agent: "Agent", street: "Street") -> None:
        """
        Create a heuristic loan_request when a regular agent is in crisis.

        The target is the wealthiest neighbor on the same street, and the
        requested amount is a modest fraction of that neighbor's wealth so the
        request is at least plausible.

        Args:
            agent: The distressed agent initiating the request.
            street: The street where the agent resides.

        Side effects:
            Sets ``agent.outbound_action`` to a loan request.
        """
        neighbors = [a for a in street.agents if a.id != agent.id]
        if not neighbors:
            return

        wealthier = [a for a in neighbors if a.wealth > agent.wealth]
        if not wealthier:
            return

        target = min(wealthier, key=lambda a: a.wealth)
        amount = max(10.0, min(100.0, target.wealth * 0.1))
        agent.outbound_action = {
            "type": "loan_request",
            "target_agent_id": target.id,
            "amount": round(amount, 2),
            "message": (
                f"I am struggling to provide for my household on Street {street.street_id}. "
                f"Could you lend me {amount:.0f} coins until things improve?"
            ),
        }

    @staticmethod
    def _cost_of_living(agent: "Agent", town_fear_index: float) -> tuple[float, bool]:
        """
        Compute a deterministic daily cost of living based on household size.

        Stage 3 panic buying: when the town fear index exceeds the panic
        threshold, agents zero out Level 3 (social) and Level 4 (esteem)
        spending and hoard Level 1 (physiological) resources, inflating that
        portion of the cost by (1 + fear_index).

        Args:
            agent: The citizen whose cost is being computed.
            town_fear_index: Current town fear index.

        Returns:
            Tuple of (daily cost, panic_mode flag).
        """
        base = max(
            5.0,
            agent.family_size * COST_PER_CAPITA
            + agent.citizen.children * COST_PER_CHILD,
        )

        panic_mode = town_fear_index > FEAR_PANIC_THRESHOLD
        if not panic_mode:
            return base, False

        # Stage 4: religious agents hoard less aggressively, retaining some trust.
        panic_multiplier = 1.0 + town_fear_index
        if getattr(agent, "religion", "Undecided") != "Undecided":
            panic_multiplier -= town_fear_index * RELIGIOUS_PANIC_DISCOUNT
            panic_multiplier = max(1.0, panic_multiplier)

        physiological = base * PHYSIOLOGICAL_COST_SHARE * panic_multiplier
        social_esteem = base * SOCIAL_ESTEEM_COST_SHARE * 0.0
        cost = max(5.0, physiological + social_esteem)
        return cost, True

    @staticmethod
    def _deterministic_event(
        agent: "Agent",
        day: int,
        income: float,
        cost: float,
        rng: random.Random | None = None,
    ) -> dict[str, Any]:
        """
        Draw a daily economic event using a locally seeded PRNG.

        This is fully reproducible: the same (agent.id, day) pair always
        produces the same outcome, making the simulation deterministic and
        debuggable while still producing rich outliers.  An external ``rng``
        can be supplied so that subsequent draws (e.g. moral anomalies) share
        the same deterministic sequence.

        Args:
            agent: The citizen experiencing the event.
            day: Current simulation day.
            income: The citizen's daily income.
            cost: The citizen's daily cost of living.
            rng: Optional pre-seeded random generator.

        Returns:
            Dictionary with ``delta`` (coin change) and ``label`` (event type).
        """
        if rng is None:
            rng = random.Random((agent.id * 100_000) + day)
        roll = rng.random()

        if roll < EVENT_WINDFALL_PROBABILITY:
            multiplier = rng.uniform(WINDFALL_MULTIPLIER_MIN, WINDFALL_MULTIPLIER_MAX)
            return {"delta": round(income * multiplier, 2), "label": "windfall"}

        if roll < EVENT_WINDFALL_PROBABILITY + EVENT_MISFORTUNE_PROBABILITY:
            multiplier = rng.uniform(MISFORTUNE_MULTIPLIER_MIN, MISFORTUNE_MULTIPLIER_MAX)
            return {"delta": round(-cost * multiplier, 2), "label": "misfortune"}

        return {"delta": 0.0, "label": "normal"}

    @staticmethod
    def _happiness_delta(
        event_label: str, net_delta: float, outliers: list[dict[str, Any]]
    ) -> float:
        """Compute a deterministic happiness adjustment for the day.

        Args:
            event_label: Classification of today's economic event.
            net_delta: Net wealth change for the agent.
            outliers: List of detected outlier events.

        Returns:
            Float happiness delta.
        """
        delta = 0.0
        if net_delta > 0:
            delta += HAPPINESS_NET_POSITIVE_BUMP
        elif net_delta < 0:
            delta += HAPPINESS_NET_NEGATIVE_BUMP

        if event_label == "windfall":
            delta += HAPPINESS_WINDFALL_BUMP
        elif event_label == "misfortune":
            delta += HAPPINESS_MISFORTUNE_BUMP
        elif event_label == "moral_anomaly":
            delta += -0.25
        elif event_label == "hidden_guilt":
            delta += -0.05

        outlier_types = {o["type"] for o in outliers}
        if "bankruptcy" in outlier_types:
            delta += HAPPINESS_BANKRUPTCY_BUMP
        if "child_starving" in outlier_types:
            delta += HAPPINESS_STARVATION_BUMP

        return delta

    @staticmethod
    def _integrity_delta(event_label: str, outliers: list[dict[str, Any]]) -> float:
        """Compute a deterministic integrity adjustment for the day.

        Args:
            event_label: Classification of today's economic event.
            outliers: List of detected outlier events.

        Returns:
            Float integrity delta.
        """
        delta = 0.0
        if event_label == "windfall":
            delta += INTEGRITY_WINDFALL_BUMP
        elif event_label == "moral_anomaly":
            delta += INTEGRITY_MORAL_ANOMALY_BUMP
        elif event_label == "hidden_guilt":
            delta += INTEGRITY_HIDDEN_GUILT_BUMP

        outlier_types = {o["type"] for o in outliers}
        if "bankruptcy" in outlier_types:
            delta += INTEGRITY_BANKRUPTCY_BUMP
        if "child_starving" in outlier_types:
            delta += INTEGRITY_STARVATION_BUMP
        if "moral_anomaly" in outlier_types:
            delta += INTEGRITY_MORAL_ANOMALY_BUMP

        return delta

    @staticmethod
    def _heuristic_narrative(
        agent: "Agent",
        income: float,
        cost: float,
        net_delta: float,
        event_label: str,
        outliers: list[dict[str, Any]],
        panic_mode: bool = False,
    ) -> tuple[str, str, str]:
        """
        Generate a short decision, reflection, and emergent action label for a
        regular citizen.  The action label is intentionally human-readable and
        feeds the bi-directional heuristic impact interpreter.

        Args:
            agent: The citizen being narrated.
            income: Daily income.
            cost: Daily cost of living.
            net_delta: Net wealth change.
            event_label: Classification of today's economic event.
            outliers: List of detected outlier events.
            panic_mode: Whether the town is in panic-buying mode.

        Returns:
            Tuple of (decision description, reflection, action_type label).
        """
        outlier_types = {o["type"] for o in outliers}

        moral_anomaly = next(
            (o for o in outliers if o["type"] == "moral_anomaly"), None
        )

        if moral_anomaly is not None:
            anomaly_type = moral_anomaly.get("anomaly_type", "wrongdoing")
            decision = (
                f"Is publicly exposed and shamed after being caught for {anomaly_type}."
            )
            action_type = "Public shame"
        elif "bankruptcy" in outlier_types:
            decision = (
                "Spends the day scrambling for shelter and food after losing "
                "the last of their savings."
            )
            action_type = "Desperate survival"
        elif "child_starving" in outlier_types:
            decision = (
                "Takes on any available labor, desperate to put food on the "
                "table for the children."
            )
            action_type = "Any labor available"
        elif "major_windfall" in outlier_types:
            decision = (
                "Celebrates cautiously and secures the unexpected fortune for "
                "the household."
            )
            action_type = "Secure windfall"
        elif "major_loss" in outlier_types:
            decision = (
                "Deals with a sudden heavy loss and tries to protect the family "
                "from its worst effects."
            )
            action_type = "Cope with loss"
        elif net_delta >= 0:
            decision = (
                "Works diligently and covers the household's daily expenses."
            )
            action_type = "Work"
        else:
            decision = (
                "Works hard but still comes up short against the day's costs."
            )
            action_type = "Work despite shortfall"

        if moral_anomaly is not None:
            anomaly_type = moral_anomaly.get("anomaly_type", "wrongdoing")
            reflection = (
                f"Humiliation burns through you as neighbors turn away; being "
                f"caught for {anomaly_type} has left your reputation in ruins."
            )
        elif "bankruptcy" in outlier_types:
            reflection = (
                "You feel the ground has vanished beneath you; every coin is "
                "gone and the future feels terrifyingly uncertain."
            )
        elif "child_starving" in outlier_types:
            reflection = (
                "Your heart breaks watching your children go without; shame "
                "and fear mix as you wonder who will help."
            )
        elif "major_windfall" in outlier_types:
            reflection = (
                "A rare surge of hope rushes through you; perhaps this windfall "
                "can finally ease the pressure on your family."
            )
        elif "major_loss" in outlier_types:
            reflection = (
                "A cold sense of setback settles over you, and you worry how "
                "many more blows the household can absorb."
            )
        elif net_delta >= 0 and event_label == "normal":
            reflection = (
                "It was an ordinary day, but you are grateful that ends met "
                "and your loved ones are safe."
            )
        elif net_delta >= 0:
            reflection = (
                "A small surplus lifts your spirits; you tuck it away with "
                "cautious optimism."
            )
        else:
            reflection = (
                "You lie awake calculating what can be cut tomorrow; the gap "
                "between income and need feels wider than ever."
            )

        if panic_mode and moral_anomaly is None:
            reflection += (
                " Panic has gripped the town; you hoard what little you can "
                "and trust no one with your remaining resources."
            )
            action_type = "Hoarding"

        return decision, reflection, action_type
