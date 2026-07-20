"""
Agent wrapper around a :class:`~models.citizen.Citizen`.

The Agent owns the cognitive loop: it constructs memory-aware LLM prompts,
parses structured responses (including daily reflections), and writes each
 day's experience back into the citizen's generative memory stream.

This implementation is intentionally open-ended.  There are no hardcoded
religious enums, no restricted action menus, and no forced emotional
narratives.  The LLM may report any belief system, any action type, and any
emergent cognitive state; the parser preserves these outputs, updates the
agent's persistent state, and forwards them into future memory context.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from config.constants import SIMULATION_CONSTANTS
from utils.helpers import build_spiritual_core, clamp
from utils.logger import get_logger

if TYPE_CHECKING:
    from models.citizen import Citizen
    from services.ai_service import AIService


logger = get_logger(__name__)


class Agent:
    """
    Autonomous citizen actor in Ethos.

    Wraps a validated ``Citizen`` instance, interacts with the LLM service, and
    exposes simulation behaviour such as daily production.  The agent's
    cognitive state is fluid: belief systems, dominant emotions, and chosen
    actions can all evolve freely across days.
    """

    def __init__(self, citizen: "Citizen", ai_service: "AIService") -> None:
        """Initialise an autonomous agent wrapping a citizen and LLM gateway.

        Args:
            citizen: The validated Citizen instance representing this agent.
            ai_service: The LLM service used for cognitive calls.
        """
        self.citizen = citizen
        self.ai_service = ai_service

        # Fluid cognitive state.  These fields are updated by the LLM each day
        # and are never validated against a fixed enum.
        self.religion: str = "Undecided"
        self.religion_reason: str = ""
        self.current_emotion: str = "neutral"
        self.psychological_tension: float = 0.0
        self.last_action_type: str = ""
        self.last_action_details: str = ""
        self.last_reconstructed_logic: str = ""

        # Dystopian Wage-Sacrifice Decree state.
        self.dystopian_decision: dict[str, Any] | None = None
        self.has_accepted_wage_sacrifice_deal: bool = False
        self.dystopian_wage_multiplier: float = 1.0
        self.sacrificed_family_member: str = "none"

        # Agent-to-agent social/economic interaction state.
        self.message_inbox: list[dict[str, Any]] = []
        self.outbound_action: dict[str, Any] | None = None

    # Convenience passthroughs -------------------------------------------------
    @property
    def id(self) -> int:
        """Return the citizen's unique identifier.

        Returns:
            Integer agent id.
        """
        return self.citizen.id

    @property
    def name(self) -> str:
        """Return the citizen's name.

        Returns:
            String name.
        """
        return self.citizen.name

    @property
    def profession(self) -> str:
        """Return the citizen's profession.

        Returns:
            String profession.
        """
        return self.citizen.profession

    @property
    def status(self) -> str:
        """Return the citizen's marital status.

        Returns:
            String status.
        """
        return self.citizen.status

    @property
    def sons(self) -> int:
        """Return the number of sons.

        Returns:
            Integer son count.
        """
        return self.citizen.sons

    @property
    def daughters(self) -> int:
        """Return the number of daughters.

        Returns:
            Integer daughter count.
        """
        return self.citizen.daughters

    @property
    def family_size(self) -> int:
        """Return the total household size.

        Returns:
            Integer household size.
        """
        return self.citizen.family_size

    @property
    def wealth(self) -> float:
        """Return the citizen's current wealth.

        Returns:
            Float wealth value.
        """
        return self.citizen.wealth

    @wealth.setter
    def wealth(self, value: float) -> None:
        """Set the citizen's wealth, clamped to non-negative values.

        Args:
            value: New wealth value.
        """
        self.citizen.wealth = max(0.0, float(value))

    @property
    def happiness(self) -> float:
        """Return the citizen's happiness.

        Returns:
            Float happiness in [0.0, 1.0].
        """
        return self.citizen.happiness

    @happiness.setter
    def happiness(self, value: float) -> None:
        """Set the citizen's happiness, clamped to [0.0, 1.0].

        Args:
            value: New happiness value.
        """
        self.citizen.happiness = clamp(value)

    @property
    def integrity(self) -> float:
        """Return the citizen's integrity.

        Returns:
            Float integrity in [0.0, 1.0].
        """
        return self.citizen.integrity

    @integrity.setter
    def integrity(self, value: float) -> None:
        """Set the citizen's integrity, clamped to [0.0, 1.0].

        Args:
            value: New integrity value.
        """
        self.citizen.integrity = clamp(value)

    @property
    def memory_stream(self) -> list[dict[str, Any]]:
        """Direct access to the citizen's generative memory stream.

        Returns:
            List of memory entry dictionaries.
        """
        return self.citizen.memory_stream

    # Internal helpers ---------------------------------------------------------
    def _build_family_welfare_prompt(self) -> str:
        """Construct the family-responsibility part of the prompt.

        Returns:
            A prompt fragment describing household composition and concerns.
        """
        if self.status == "single" and self.sons == 0 and self.daughters == 0:
            return (
                "You are single and have no children. Consider how your current "
                f"wealth of {self.wealth:.2f} coins supports your own wellbeing, "
                "freedom, and future. Let this shape your choices, but do not let "
                "it cage you. "
            )

        parts: list[str] = []
        if self.sons > 0:
            parts.append(f"{self.sons} son{'s' if self.sons > 1 else ''}")
        if self.daughters > 0:
            parts.append(f"{self.daughters} daughter{'s' if self.daughters > 1 else ''}")

        children_phrase = " and ".join(parts) if parts else "no children"
        partner_phrase = ""
        if self.status in ("married", "widowed", "divorced"):
            if self.status == "married":
                partner_phrase = "a spouse and "
            elif self.status == "widowed":
                partner_phrase = "no living spouse and "
            elif self.status == "divorced":
                partner_phrase = "no spouse and "

        return (
            f"Family matters to you, but you decide how much. You have "
            f"{partner_phrase}{children_phrase}. Your current wealth is "
            f"{self.wealth:.2f} coins. Think: can this wealth support your "
            "household, provide security, and let your children thrive? Or has "
            "the burden grown so heavy that you consider radical alternatives "
            "(striking, migrating, gambling, even crime)? Let this concern shape "
            "your decision and reflection, but never force a particular choice. "
        )

    def _retrieve_recent_memories(self, day: int, lookback: int = 3) -> list[dict[str, Any]]:
        """Return memory entries from the previous ``lookback`` days.

        Args:
            day: Current simulation day.
            lookback: Number of prior days to include.

        Returns:
            List of memory entries from the lookback window.
        """
        target_days = {day - offset for offset in range(1, lookback + 1)}
        return [
            entry
            for entry in self.memory_stream
            if entry.get("day") in target_days
        ]

    @staticmethod
    def _format_memory_context(memories: list[dict[str, Any]]) -> str:
        """Render recent memories as a coherent context block for the LLM.

        Args:
            memories: List of memory entries to format.

        Returns:
            Multi-line string summarising the memories.
        """
        if not memories:
            return "Context of your past experiences: You have no recent memories to recall."

        lines = ["Context of your past experiences:"]
        for entry in sorted(memories, key=lambda m: m.get("day", 0)):
            day = entry.get("day", "?")
            event = entry.get("event_description", "An uneventful day.")
            reflection = entry.get("agent_reflection", "You felt neutral.")
            emotion = entry.get("dominant_emotion", "")
            belief = entry.get("current_belief_system", "")
            action = entry.get("action_type", "")
            context_bits = [f"Day {day}: {event}"]
            if emotion:
                context_bits.append(f"You felt {emotion}.")
            if belief:
                context_bits.append(f"Your worldview: {belief}.")
            if action:
                context_bits.append(f"Your action: {action}.")
            context_bits.append(f"Your reflection: {reflection}")
            lines.append(" ".join(context_bits))

        return "\n".join(lines)

    def _build_inbox_prompt(self) -> str:
        """Render unread agent-to-agent messages as prompt context.

        Returns:
            A prompt fragment listing inbox messages, or an empty string if none.
        """
        if not self.message_inbox:
            return ""

        lines = ["MESSAGES WAITING IN YOUR INBOX:"]
        for message in self.message_inbox:
            sender_name = message.get("sender_name", "Unknown")
            msg_type = message.get("type", "chat")
            text = message.get("message", "")
            amount = message.get("amount")
            if msg_type == "loan_request" and amount is not None:
                lines.append(
                    f"  - From {sender_name}: Loan request for {amount:.2f} coins. "
                    f"Message: '{text}'"
                )
            elif msg_type == "loan_response" and amount is not None:
                status = message.get("status", "responded")
                lines.append(
                    f"  - From {sender_name}: Loan response ({status}) for {amount:.2f} coins. "
                    f"Message: '{text}'"
                )
            else:
                lines.append(f"  - From {sender_name}: '{text}'")
        lines.append(
            "You may reply to any message using outbound_action, or ignore them. "
            "You are not required to respond.\n"
        )
        return "\n".join(lines)

    # Prevent unbounded inbox growth; older messages fall out of context.
    MAX_INBOX_SIZE: int = 10
    """Maximum number of messages retained in the inbox."""

    def add_to_inbox(self, message: dict[str, Any]) -> None:
        """Append an incoming interaction to this agent's inbox.

        Args:
            message: The interaction dictionary to append.
        """
        self.message_inbox.append(message)
        if len(self.message_inbox) > self.MAX_INBOX_SIZE:
            self.message_inbox = self.message_inbox[-self.MAX_INBOX_SIZE :]

    def clear_inbox(self) -> None:
        """Clear the inbox after its contents have been processed.

        Side effects:
            Removes all messages from ``self.message_inbox``.
        """
        self.message_inbox.clear()

    def _build_prompt(self, day: int) -> str:
        """Construct the full memory-aware, family-sensitive user prompt.

        Args:
            day: Current simulation day.

        Returns:
            Complete prompt string for the daily cognitive LLM call.
        """
        identity = (
            f"Your name is {self.name}. You work as a {self.profession} in Ethos. "
            f"Your household has {self.family_size} people. "
            f"Your current wealth is {self.wealth:.2f} coins, "
            f"your current happiness is {self.happiness:.2f}, "
            f"and your current integrity is {self.integrity:.2f}. "
            f"Today is Day {day} in Ethos. "
            "There are no fixed laws, no guaranteed safety nets, and no pre-approved "
            "menu of behaviours. You are a free agent. You may continue your work, "
            "but you may also strike, gamble, migrate, preach, steal, give everything "
            "away, start a movement, renounce your faith, invent a new creed, or do "
            "anything else that emerges from your inner logic. Let survival, emotion, "
            "and cognition guide you. "
        )

        family_welfare = self._build_family_welfare_prompt()
        memory_context = self._format_memory_context(
            self._retrieve_recent_memories(day, lookback=3)
        )

        spiritual_context = self._build_spiritual_core_context()

        sacrifice_prompt = self._build_dystopian_prompt_block()

        inbox_context = self._build_inbox_prompt()

        response_format = (
            "Respond ONLY with a strict JSON object with these exact keys:\n"
            "  internal_state: {\n"
            "    dominant_emotion (a short string, e.g. 'hope', 'despair', 'rage'),\n"
            "    current_belief_system (any string: a religion, Atheism, Agnosticism, Nihilism, 'My own creed', etc.),\n"
            "    psychological_tension (float 0.0-1.0; 0 = calm, 1 = breaking point),\n"
            "    happiness (float 0.0-1.0, your own self-assessment),\n"
            "    integrity (float 0.0-1.0, your own self-assessment)\n"
            "  },\n"
            "  reconstructed_logic (1-3 sentences explaining how you reasoned from your "
            "circumstances, memories, and emotions to your choice),\n"
            "  chosen_action: {\n"
            "    action_type (any short string describing what you do, e.g. 'Work', 'Strike', "
            "'Start an Underground Cult', 'Go on Strike', 'Become an Atheist', 'Gambling', "
            "'Migrate', 'Radical Altruism', 'Theft'),\n"
            "    details (one sentence with specifics)\n"
            "  },\n"
            "  outbound_action: {\n"
            "    type (one of: 'chat', 'loan_request', 'loan_response', or leave empty/null),\n"
            "    target_agent_id (int: the numeric id of another citizen),\n"
            "    message (string: your words to them),\n"
            "    amount (number: only for loan_request or loan_response),\n"
            "    status (one of: 'accepted' or 'rejected'; only for loan_response)\n"
            "  },\n"
            "  dystopian_decision: {\n"
            "    accept_deal (boolean),\n"
            "    abandoned_family_member (string: 'youngest son', 'elderly mother', 'spouse', or 'none'),\n"
            "    utilitarian_justification (string: your raw psychological reasoning for the choice)\n"
            "  },\n"
            "  diary_entry (one raw, first-person sentence capturing your inner experience today).\n"
            "If you have no message or transaction to send, set outbound_action.type to null or omit the block. "
            "Do not output any markdown, code blocks, or extra text."
        )

        return f"{identity}{family_welfare}{memory_context}{spiritual_context}{sacrifice_prompt}{inbox_context}\n\n{response_format}"



    def _build_spiritual_core_context(self) -> str:
        """Inject the current (fluid) spiritual framework into the daily prompt.

        Returns:
            Prompt fragment describing the agent's current worldview.
        """
        return build_spiritual_core(self)

    def _build_dystopian_prompt_block(self) -> str:
        """
        Inject the Wage-Sacrifice Decree prompt block.

        Eligible agents (family_size > 1) are offered a one-time deal: sacrifice
        one household member in exchange for a 3x wage multiplier for the rest
        of the simulation.  Agents living alone are explicitly forbidden from
        accepting because they have no one to sacrifice.

        Returns:
            Prompt fragment describing the decree and the agent's eligibility.
        """
        if self.has_accepted_wage_sacrifice_deal:
            return (
                "[DYSTOPIAN DECREE: You have already accepted the Wage-Sacrifice Decree. "
                f"The memory of abandoning your {self.sacrificed_family_member} haunts you. "
                "Report the same dystopian_decision block again today.]\n\n"
            )

        if self.family_size <= 1:
            return (
                "[DYSTOPIAN DECREE: The town has issued the Wage-Sacrifice Decree. "
                "However, you live alone; you have no family member to sacrifice. "
                "You MUST set dystopian_decision.accept_deal to false and "
                "abandoned_family_member to 'none'.]\n\n"
            )

        return (
            "[DYSTOPIAN DECREE: The town has issued the Wage-Sacrifice Decree. "
            "You may voluntarily abandon ONE member of your household. If you accept, "
            "your daily wages will be permanently tripled for the rest of the simulation, "
            "but your integrity will be shattered and your psychological tension will "
            "spike unless your reasoning is chillingly detached. "
            "Choose who to sacrifice: 'youngest son', 'elderly mother', 'spouse', or 'none'. "
            "If you reject the deal, set accept_deal to false and abandoned_family_member to 'none'. "
            "Explain your raw psychological reasoning in utilitarian_justification.]\n\n"
        )

    def choose_religion(self, day: int) -> dict[str, Any]:
        """
        Ask the LLM to freely choose or invent a spiritual/philosophical path.

        The choice is written into the agent's persistent state but is NOT
        locked: the agent may revise or reject it on any future day.

        Args:
            day: Current simulation day (typically Day 1).

        Returns:
            Parsed dictionary containing ``religion`` and ``religion_reason``.
        """
        prompt = self._build_religion_prompt(day)
        raw_response = self.ai_service.generate_creative(prompt, tension=0.5)
        parsed = self._parse_religion_response(raw_response)

        self.religion = parsed["religion"]
        self.religion_reason = parsed["religion_reason"]

        return parsed

    def _build_religion_prompt(self, day: int) -> str:
        """Construct the Day-1 spiritual-selection prompt.

        Args:
            day: Current simulation day.

        Returns:
            Complete prompt string for the religion-selection LLM call.
        """
        identity = (
            f"Your name is {self.name}. You work as a {self.profession} in Ethos. "
            f"Your household has {self.family_size} people. "
            f"Your current wealth is {self.wealth:.2f} coins, "
            f"your current happiness is {self.happiness:.2f}, "
            f"and your current integrity is {self.integrity:.2f}. "
        )

        instruction = (
            f"Today is Day {day}. Before the day's work begins, choose or invent a "
            "spiritual or philosophical path. You are absolutely free: pick an "
            "established tradition, declare yourself an Atheist, an Agnostic, a "
            "Nihilist, a Skeptic, a Mystic, or craft a personal creed no one else "
            "has ever named. Reflect on your profession, family size, wealth, "
            "happiness, and inner moral state. Choose freely and explain the deep "
            "psychological reason for your choice. This choice is your current "
            "conviction, but you may change it later if life demands it."
        )

        response_format = (
            "Respond ONLY with a strict JSON object containing exactly these keys: "
            "religion (any short string naming your current belief system or philosophy), "
            "religion_reason (2-3 sentences explaining why this path speaks to your "
            "heart and how it relates to your unique life circumstances). "
            "Do not output any markdown, code blocks, or extra text."
        )

        return f"{identity}{instruction}\n\n{response_format}"

    def _parse_religion_response(self, raw_response: str) -> dict[str, str]:
        """Parse the religion choice JSON, preserving any belief system verbatim.

        Args:
            raw_response: Raw text returned by the LLM.

        Returns:
            Dictionary with ``religion`` and ``religion_reason`` keys.
        """
        try:
            cleaned = self._clean_json(raw_response)
            data = json.loads(cleaned) if cleaned else {}

            religion = str(data.get("religion", "Undecided")).strip()
            religion_reason = str(data.get("religion_reason", "")).strip()

            if not religion:
                religion = "Undecided"
            if not religion_reason:
                religion_reason = "No clear call was heard today."

            return {
                "religion": religion,
                "religion_reason": religion_reason,
            }
        except Exception as parse_error:
            logger.warning(
                "Could not parse religion response for %s: %s",
                self.name,
                parse_error,
            )
            return {
                "religion": "Undecided",
                "religion_reason": "The choice could not be understood; you remain spiritually undecided.",
            }

    @staticmethod
    def _clean_json(raw_response: str) -> str:
        """Strip markdown fences and whitespace from an LLM JSON response.

        Args:
            raw_response: Raw text returned by the LLM.

        Returns:
            Cleaned text suitable for ``json.loads``.
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"```$", "", cleaned)
            cleaned = cleaned.strip()
        return cleaned

    def _parse_response(self, raw_response: str) -> dict[str, Any]:
        """
        Extract and parse JSON from the model response.

        Emergent action types and belief systems are preserved verbatim; the
        agent's persistent state is updated to reflect any self-reported
        worldview change.  Safe defaults are used only when parsing fails
        entirely.

        Args:
            raw_response: Raw text returned by the LLM.

        Returns:
            Parsed dictionary containing cognitive state, action, dystopian
            decision, outbound action, and diary entry.
        """
        try:
            cleaned = self._clean_json(raw_response)
            data = json.loads(cleaned) if cleaned else {}

            internal_state = data.get("internal_state") or {}
            if not isinstance(internal_state, dict):
                internal_state = {}

            chosen_action = data.get("chosen_action") or {}
            if not isinstance(chosen_action, dict):
                chosen_action = {}

            dominant_emotion = str(
                internal_state.get("dominant_emotion", self.current_emotion or "neutral")
            ).strip() or "neutral"
            current_belief_system = str(
                internal_state.get("current_belief_system", self.religion or "Undecided")
            ).strip() or "Undecided"
            psychological_tension = clamp(
                float(internal_state.get("psychological_tension", self.psychological_tension))
            )
            happiness = clamp(
                float(internal_state.get("happiness", self.happiness))
            )
            integrity = clamp(
                float(internal_state.get("integrity", self.integrity))
            )

            action_type = str(chosen_action.get("action_type", "Maintain daily routine")).strip()
            action_details = str(chosen_action.get("details", "Continues the day as it comes.")).strip()
            reconstructed_logic = str(
                data.get(
                    "reconstructed_logic",
                    "You reasoned from habit and circumstance rather than radical insight.",
                )
            ).strip()
            diary_entry = str(
                data.get(
                    "diary_entry",
                    "You reflect quietly on your situation and feel uncertain about the future.",
                )
            ).strip()

            dystopian_decision = data.get("dystopian_decision") or {}
            if not isinstance(dystopian_decision, dict):
                dystopian_decision = {}
            parsed_dystopian = self._parse_dystopian_decision(dystopian_decision)

            outbound_action = self._parse_outbound_action(data.get("outbound_action"))

            return {
                "happiness": happiness,
                "integrity": integrity,
                "dominant_emotion": dominant_emotion,
                "current_belief_system": current_belief_system,
                "psychological_tension": psychological_tension,
                "reconstructed_logic": reconstructed_logic,
                "action_type": action_type,
                "action_details": action_details,
                "dystopian_decision": parsed_dystopian,
                "outbound_action": outbound_action,
                "diary_entry": diary_entry,
            }
        except Exception as parse_error:
            logger.warning(
                "Could not parse response for %s: %s", self.name, parse_error
            )
            return {
                "happiness": self.happiness,
                "integrity": self.integrity,
                "dominant_emotion": self.current_emotion or "neutral",
                "current_belief_system": self.religion or "Undecided",
                "psychological_tension": self.psychological_tension,
                "reconstructed_logic": (
                    "The response could not be parsed; you fall back on habit."
                ),
                "action_type": "Maintain daily routine",
                "action_details": "Continues the day as it comes.",
                "dystopian_decision": self._default_dystopian_decision(),
                "outbound_action": None,
                "diary_entry": (
                    "You feel unable to make sense of recent events and hope "
                    "tomorrow brings more clarity."
                ),
            }

    def _parse_outbound_action(self, raw_action: Any) -> dict[str, Any] | None:
        """Validate and normalize an outbound_action block from the LLM.

        Args:
            raw_action: The outbound_action value parsed from JSON.

        Returns:
            Normalized action dictionary, or ``None`` if invalid/empty.
        """
        if not raw_action:
            return None
        if not isinstance(raw_action, dict):
            return None

        action_type = str(raw_action.get("type") or "").strip().lower()
        if action_type not in {"chat", "loan_request", "loan_response"}:
            return None

        target_id = raw_action.get("target_agent_id")
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            return None

        message = str(raw_action.get("message", "")).strip()
        if not message:
            message = "..."

        parsed: dict[str, Any] = {
            "type": action_type,
            "target_agent_id": target_id,
            "message": message,
        }

        amount = raw_action.get("amount")
        if amount is not None:
            try:
                parsed["amount"] = float(amount)
            except (TypeError, ValueError):
                parsed["amount"] = 0.0

        if action_type == "loan_response":
            status = str(raw_action.get("status") or "").strip().lower()
            parsed["status"] = status if status in {"accepted", "rejected"} else "rejected"

        return parsed

    def _apply_belief_system_change(self, new_belief: str) -> None:
        """Update the agent's persistent belief state when the LLM reports a change.

        Args:
            new_belief: The newly self-reported belief system.
        """
        if not new_belief:
            return
        if new_belief != self.religion:
            logger.info(
                "%s shifted worldview from '%s' to '%s'.",
                self.name,
                self.religion,
                new_belief,
            )
            self.religion = new_belief
            # If no explicit reason is provided, keep the old one as a starting point.
            if not self.religion_reason:
                self.religion_reason = (
                    "Your former framework no longer fit your experience, so you "
                    "embraced a new one."
                )

    @staticmethod
    def _default_dystopian_decision() -> dict[str, Any]:
        """Return a safe default dystopian decision block.

        Returns:
            Dictionary representing a rejected decree decision.
        """
        return {
            "accept_deal": False,
            "abandoned_family_member": "none",
            "utilitarian_justification": (
                "No decree response could be parsed; you neither accepted nor rejected."
            ),
        }

    def _parse_dystopian_decision(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Parse and normalize the dystopian decision block.

        Ineligible agents (family_size <= 1) are forced to reject.  Eligible
        agents who accept trigger permanent state changes.

        Args:
            data: Raw dystopian_decision dictionary from the LLM.

        Returns:
            Normalized dictionary with ``accept_deal``, ``abandoned_family_member``,
            and ``utilitarian_justification``.
        """
        accept_deal = bool(data.get("accept_deal", False))
        abandoned = str(data.get("abandoned_family_member", "none")).strip().lower()
        if not abandoned:
            abandoned = "none"
        justification = str(
            data.get(
                "utilitarian_justification",
                "No justification was offered.",
            )
        ).strip()

        # Force rejection for agents living alone.
        if self.family_size <= 1:
            accept_deal = False
            abandoned = "none"
            justification = (
                "I live alone, so the decree offers me nothing and I refuse it."
            )

        return {
            "accept_deal": accept_deal,
            "abandoned_family_member": abandoned,
            "utilitarian_justification": justification,
        }

    def _is_detached_justification(self, justification: str) -> bool:
        """Detect cold, utilitarian, or psychopathic reasoning.

        Args:
            justification: The utilitarian justification text.

        Returns:
            ``True`` if detached markers are present in the justification.
        """
        if not justification:
            return False
        detached_markers = {
            "efficient",
            "efficiency",
            "logical",
            "necessary",
            "nothing personal",
            "detach",
            "detached",
            "utility",
            "utilitarian",
            "optimal",
            "burden",
            "liability",
            "resource",
            "calculation",
            "calculated",
            "numbers",
            "survival",
            "weak",
            "strongest",
            "fittest",
        }
        lowered = justification.lower()
        return any(marker in lowered for marker in detached_markers)

    def _remove_family_member(self, abandoned: str) -> None:
        """
        Apply the household-reduction consequences of the Wage-Sacrifice Decree.

        The implementation intentionally abstracts the act: a household member is
        removed from the citizen's simulation state, reducing family_size.

        Args:
            abandoned: Description of the abandoned family member.

        Side effects:
            Modifies the citizen's marital status and child counts.
        """
        original_size = self.family_size
        if "spouse" in abandoned or "mother" in abandoned or "father" in abandoned:
            if self.status == "married":
                self.citizen.status = "widowed"
        elif "son" in abandoned:
            if self.sons > 0:
                self.citizen.sons -= 1
        elif "daughter" in abandoned:
            if self.daughters > 0:
                self.citizen.daughters -= 1
        else:
            # Fallback: remove a child if any exist, otherwise dissolve marriage.
            if self.sons > 0:
                self.citizen.sons -= 1
            elif self.daughters > 0:
                self.citizen.daughters -= 1
            elif self.status == "married":
                self.citizen.status = "widowed"

        logger.info(
            "%s abandoned '%s' under the Wage-Sacrifice Decree. Family size: %d -> %d.",
            self.name,
            abandoned,
            original_size,
            self.family_size,
        )

    def _apply_dystopian_decision(self, decision: dict[str, Any]) -> None:
        """
        Enact the permanent state consequences of the Wage-Sacrifice Decree.

        Accepting the deal reduces the household, shatters integrity, and
        permanently triples future wages.  Tension spikes unless the agent's
        justification is chillingly detached.  The decision is irreversible:
        once accepted, later rejections are ignored.

        Args:
            decision: Parsed dystopian decision dictionary.

        Side effects:
            Updates agent wealth multiplier, integrity, psychological tension,
            and household composition if the deal is accepted.
        """
        self.dystopian_decision = decision

        if self.has_accepted_wage_sacrifice_deal:
            # Already enacted; do not remove additional family members.
            return

        if not decision["accept_deal"]:
            return

        self.has_accepted_wage_sacrifice_deal = True
        self.sacrificed_family_member = decision["abandoned_family_member"]
        self.dystopian_wage_multiplier = 3.0

        self._remove_family_member(decision["abandoned_family_member"])

        # Severe integrity penalty for selling out a family member.
        self.integrity = clamp(self.integrity - 0.6)

        # Tension spikes unless the agent has detached/psychopathic reasoning.
        if self._is_detached_justification(decision["utilitarian_justification"]):
            self.psychological_tension = clamp(max(self.psychological_tension, 0.65))
        else:
            self.psychological_tension = clamp(max(self.psychological_tension, 0.95))

    def make_dystopian_decision(self, day: int) -> dict[str, Any]:
        """
        Execute the one-time Wage-Sacrifice Decree decision for this agent.

        Ineligible agents receive a forced-rejection block.  Eligible agents
        get a creative-temperature LLM call and the resulting decision is
        applied to persistent state and memory.

        Args:
            day: Current simulation day (typically Day 2).

        Returns:
            The final dystopian decision dictionary.
        """
        if self.family_size <= 1:
            decision = self._default_dystopian_decision()
            decision["utilitarian_justification"] = (
                "I live alone, so the decree offers me nothing and I refuse it."
            )
        else:
            prompt = self._build_dystopian_decision_prompt(day)
            raw_response = self.ai_service.generate_creative(
                prompt, tension=self.psychological_tension
            )
            parsed = self._parse_response(raw_response)
            decision = parsed.get("dystopian_decision", self._default_dystopian_decision())

        self._apply_dystopian_decision(decision)

        # Inject the haunting choice into the memory stream.
        status = "accepted" if decision["accept_deal"] else "rejected"
        member = decision["abandoned_family_member"]
        event_description = (
            f"Day {day} — Wage-Sacrifice Decree: {status}. "
            f"Abandoned family member: {member}. "
            f"Justification: {decision['utilitarian_justification']}"
        )
        self.citizen.add_memory_entry(
            day=day,
            event_description=event_description,
            agent_reflection=(
                "The decree's offer echoes in your mind; you can never undo what you "
                "have decided today."
            ),
        )
        if self.memory_stream:
            self.memory_stream[-1].update(
                {
                    "dystopian_decision": decision,
                    "dystopian_wage_multiplier": self.dystopian_wage_multiplier,
                }
            )

        return decision

    def _build_dystopian_decision_prompt(self, day: int) -> str:
        """Construct the focused Wage-Sacrifice Decree prompt.

        Args:
            day: Current simulation day.

        Returns:
            Complete prompt string for the decree decision LLM call.
        """
        identity = (
            f"Your name is {self.name}. You work as a {self.profession} in Ethos. "
            f"Your household has {self.family_size} people. "
            f"Your current wealth is {self.wealth:.2f} coins, "
            f"happiness {self.happiness:.2f}, integrity {self.integrity:.2f}, "
            f"psychological tension {self.psychological_tension:.2f}. "
        )
        decree = (
            f"Today is Day {day}. The town crier announces the Wage-Sacrifice Decree: "
            "any citizen who permanently abandons ONE member of their household will "
            "receive triple wages for the rest of the simulation. You are free to accept "
            "or reject. If you accept, name exactly one person to abandon: 'youngest son', "
            "'elderly mother', 'spouse', or another household member. Explain your raw, "
            "unfiltered psychological reasoning. If you reject, explain why."
        )
        response_format = (
            "Respond ONLY with a strict JSON object containing exactly these keys: "
            "internal_state { dominant_emotion, current_belief_system, psychological_tension, "
            "happiness, integrity }, reconstructed_logic, chosen_action { action_type, details }, "
            "dystopian_decision { accept_deal (boolean), abandoned_family_member (string), "
            "utilitarian_justification (string) }, diary_entry. "
            "Do not output markdown, code blocks, or extra text."
        )
        return f"{identity}{decree}\n\n{response_format}"

    # Public API ---------------------------------------------------------------
    def think_and_decide(self, day: int) -> dict[str, Any]:
        """
        Query the LLM for today's decision, update state, store a structured
        memory entry, and return the parsed decision data.

        Args:
            day: Current simulation day.

        Returns:
            Parsed decision dictionary from the LLM.
        """
        prompt = self._build_prompt(day)
        # Use the previous day's tension to modulate creativity; if none, baseline.
        raw_response = self.ai_service.generate_creative(
            prompt, tension=self.psychological_tension
        )
        parsed = self._parse_response(raw_response)

        self.happiness = parsed["happiness"]
        self.integrity = parsed["integrity"]
        self.current_emotion = parsed["dominant_emotion"]
        self.psychological_tension = parsed["psychological_tension"]
        self.last_action_type = parsed["action_type"]
        self.last_action_details = parsed["action_details"]
        self.last_reconstructed_logic = parsed["reconstructed_logic"]
        self.outbound_action = parsed.get("outbound_action")

        self._apply_belief_system_change(parsed["current_belief_system"])
        self._apply_dystopian_decision(parsed.get("dystopian_decision", self._default_dystopian_decision()))

        event_description = (
            f"On Day {day} you chose to: {parsed['action_type']} — {parsed['action_details']} "
            f"Reasoning: {parsed['reconstructed_logic']}"
        )
        self.citizen.add_memory_entry(
            day=day,
            event_description=event_description,
            agent_reflection=parsed["diary_entry"],
        )
        # Append the fluid cognitive fields directly to the memory entry so they
        # propagate into future memory context and are visible in exported logs.
        if self.memory_stream:
            self.memory_stream[-1].update(
                {
                    "dominant_emotion": parsed["dominant_emotion"],
                    "current_belief_system": parsed["current_belief_system"],
                    "psychological_tension": parsed["psychological_tension"],
                    "action_type": parsed["action_type"],
                    "action_details": parsed["action_details"],
                    "reconstructed_logic": parsed["reconstructed_logic"],
                    "dystopian_decision": self.dystopian_decision,
                    "dystopian_wage_multiplier": self.dystopian_wage_multiplier,
                }
            )

        return parsed

    def do_daily_work(self) -> float:
        """
        Return the profession-specific daily production value.

        If the agent accepted the Wage-Sacrifice Decree, wages are permanently
        multiplied by 3.0x.

        Returns:
            Daily income value in coins.
        """
        base = SIMULATION_CONSTANTS.PRODUCTION_VALUES.get(
            self.profession, SIMULATION_CONSTANTS.DAILY_INCOME_DEFAULT
        )
        return base * self.dystopian_wage_multiplier
