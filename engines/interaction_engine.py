"""
Agent-to-agent social and economic interaction router.

This engine implements an asynchronous Inbox/Router pattern.  Each day, after
all agents have completed their cognitive step, the engine:

1. Collects any ``outbound_action`` produced by agents (chat, loan_request,
   loan_response).
2. Routes those actions to the target agent's ``message_inbox``.
3. Executes accepted loans by transferring wealth between citizens.
4. Generates a small number of random same-street encounters to keep the city
   feeling alive without requiring O(N^2) API calls.

All interactions are returned as structured records that callers can persist
for analysis.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from utils.logger import get_logger

if TYPE_CHECKING:
    from models.agent import Agent
    from models.city import City

logger = get_logger(__name__)


class InteractionEngine:
    """Routes agent-to-agent messages and financial transactions."""

    def __init__(self) -> None:
        pass

    def process_day(self, city: "City", day: int) -> list[dict[str, Any]]:
        """Process outbound actions and random encounters for the given day."""
        interactions: list[dict[str, Any]] = []
        interactions.extend(self.process_outbound_actions(city, day))
        interactions.extend(self.generate_random_encounters(city, day))
        return interactions

    def process_outbound_actions(self, city: "City", day: int) -> list[dict[str, Any]]:
        """
        Route every pending outbound_action to its target agent.

        Actions are cleared after processing so they are not replayed on a
        subsequent day.
        """
        interactions: list[dict[str, Any]] = []
        for agent in list(city.agents.values()):
            action = getattr(agent, "outbound_action", None)
            if not action:
                continue
            try:
                interaction = self._route_action(city, agent, action, day)
                if interaction:
                    interactions.append(interaction)
            except Exception as error:
                logger.warning(
                    "Interaction routing failed for %s (id=%s): %s",
                    agent.name,
                    agent.id,
                    error,
                )
            finally:
                agent.outbound_action = None
        return interactions

    def _route_action(
        self,
        city: "City",
        sender: "Agent",
        action: dict[str, Any],
        day: int,
    ) -> dict[str, Any] | None:
        """Dispatch a single outbound action by its type."""
        target_id = action.get("target_agent_id")
        target = city.agents.get(target_id)
        if target is None or target.id == sender.id:
            logger.debug(
                "Invalid target %s for outbound action from agent %s.",
                target_id,
                sender.id,
            )
            return None

        action_type = str(action.get("type", "")).strip().lower()
        message = str(action.get("message", "")).strip() or "..."
        amount = action.get("amount", 0.0)
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = 0.0

        if action_type == "chat":
            return self._send_chat(sender, target, message, day)
        if action_type == "loan_request":
            return self._send_loan_request(sender, target, amount, message, day)
        if action_type == "loan_response":
            status = str(action.get("status") or "rejected").strip().lower()
            return self._send_loan_response(
                sender, target, amount, status, message, day
            )

        logger.debug("Unknown outbound action type '%s' from agent %s.", action_type, sender.id)
        return None

    def _send_chat(
        self,
        sender: "Agent",
        target: "Agent",
        message: str,
        day: int,
    ) -> dict[str, Any]:
        """Deliver a chat message to the target's inbox."""
        target.add_to_inbox(
            {
                "day": day,
                "sender_id": sender.id,
                "sender_name": sender.name,
                "type": "chat",
                "message": message,
            }
        )
        return {
            "day": day,
            "sender_id": sender.id,
            "receiver_id": target.id,
            "interaction_type": "chat",
            "amount": 0.0,
            "message": message,
        }

    def _send_loan_request(
        self,
        sender: "Agent",
        target: "Agent",
        amount: float,
        message: str,
        day: int,
    ) -> dict[str, Any]:
        """Deliver a loan request to the target's inbox."""
        safe_amount = max(0.0, float(amount))
        target.add_to_inbox(
            {
                "day": day,
                "sender_id": sender.id,
                "sender_name": sender.name,
                "type": "loan_request",
                "amount": safe_amount,
                "message": message,
            }
        )
        return {
            "day": day,
            "sender_id": sender.id,
            "receiver_id": target.id,
            "interaction_type": "loan_request",
            "amount": safe_amount,
            "message": message,
        }

    def _send_loan_response(
        self,
        sender: "Agent",
        target: "Agent",
        amount: float,
        status: str,
        message: str,
        day: int,
    ) -> dict[str, Any]:
        """
        Deliver a loan response and execute an accepted transfer.

        The ``sender`` is the citizen responding to a previous loan request; if
        they accept, they act as the lender and wealth moves from them to the
        original requester (``target``).
        """
        safe_amount = max(0.0, float(amount))
        accepted = status == "accepted"
        if accepted:
            self._transfer_wealth(sender, target, safe_amount)

        target.add_to_inbox(
            {
                "day": day,
                "sender_id": sender.id,
                "sender_name": sender.name,
                "type": "loan_response",
                "amount": safe_amount,
                "status": status,
                "message": message,
            }
        )
        return {
            "day": day,
            "sender_id": sender.id,
            "receiver_id": target.id,
            "interaction_type": f"loan_response_{status}",
            "amount": safe_amount,
            "message": message,
        }

    @staticmethod
    def _transfer_wealth(lender: "Agent", borrower: "Agent", amount: float) -> float:
        """Move coins from lender to borrower, capped by available wealth."""
        safe_amount = max(0.0, float(amount))
        if safe_amount <= 0:
            return 0.0
        transferable = min(safe_amount, lender.wealth)
        if transferable <= 0:
            logger.info(
                "Loan from %s to %s could not be funded (lender has %.2f coins).",
                lender.name,
                borrower.name,
                lender.wealth,
            )
            return 0.0

        lender.wealth -= transferable
        borrower.wealth += transferable
        logger.info(
            "Loan executed: %s lent %.2f coins to %s.",
            lender.name,
            transferable,
            borrower.name,
        )
        return transferable

    def generate_random_encounters(self, city: "City", day: int) -> list[dict[str, Any]]:
        """
        Create lightweight same-street social encounters.

        Each street has a small chance of producing one random chat between two
        distinct agents.  These encounters do not invoke the LLM; they are
        generated heuristically to enrich the social graph.
        """
        interactions: list[dict[str, Any]] = []
        for street in city.streets:
            if len(street.agents) < 2:
                continue
            # Deterministic local RNG keyed by street and day so the simulation
            # remains fully reproducible.
            rng = random.Random((street.street_id * 10_000) + day)
            if rng.random() >= 0.35:
                continue

            a, b = rng.sample(street.agents, 2)
            topic = rng.choice(
                [
                    "greeted",
                    "shared a rumor about the town crier",
                    "complained about the cost of bread",
                    "exchanged news from the market",
                    "whispered about a neighbor's good fortune",
                    "argued quietly over street politics",
                ]
            )
            message = f"{a.name} {topic} with {b.name} on Street {street.street_id}."
            a.add_to_inbox(
                {
                    "day": day,
                    "sender_id": b.id,
                    "sender_name": b.name,
                    "type": "chat",
                    "message": message,
                }
            )
            interactions.append(
                {
                    "day": day,
                    "sender_id": a.id,
                    "receiver_id": b.id,
                    "interaction_type": "random_encounter",
                    "amount": 0.0,
                    "message": message,
                }
            )
        return interactions
