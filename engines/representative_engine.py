"""
Street representative logic: zoning, telemetry, LLM delegation, and parsing.

This engine owns the socio-economic street partitioning, the election of one
representative per street, the construction of the Street Telemetry Report,
and the single daily LLM call that lets each representative reflect on their
own household and their neighbors.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from config.constants import SIMULATION_CONSTANTS
from engines.constants import (
    AGENTS_PER_STREET,
    CIVIC_PROFESSIONS,
    EXPECTED_POPULATION,
    FEAR_PANIC_THRESHOLD,
    REPRESENTATIVE_SYSTEM_PROMPT,
    STREET_CLASS_LABELS,
    STREET_COUNT,
)
from models.street import Street
from utils.helpers import build_spiritual_core, clamp, interpret_action_type
from utils.logger import get_logger

if TYPE_CHECKING:
    from models.agent import Agent
    from services.ai_service import AIService

logger = get_logger(__name__)


class RepresentativeEngine:
    """
    Engine responsible for street zoning, telemetry aggregation, and the
    per-street representative LLM cognitive loop.
    """

    def __init__(self, ai_service: "AIService" | None = None) -> None:
        self.ai_service = ai_service

    def organize_streets(self, agents: dict[int, "Agent"]) -> list[Street]:
        """
        Sort all agents by ascending wealth and descending family burden, then
        partition them into exactly 10 streets of 10 agents each.

        Streets 1-2 are the vulnerable class, 5-6 the middle class, and 9-10
        the financial elite.  Each street elects one representative based on a
        combination of high integrity and civic profession status.
        """
        population = len(agents)
        if population != EXPECTED_POPULATION:
            raise ValueError(
                f"Expected exactly {EXPECTED_POPULATION} citizens for street zoning, "
                f"got {population}."
            )

        sorted_agents = sorted(
            agents.values(),
            key=lambda agent: (
                agent.wealth,
                -agent.family_size,
                -agent.citizen.children,
                agent.id,
            ),
        )

        streets: list[Street] = []
        for idx in range(STREET_COUNT):
            start = idx * AGENTS_PER_STREET
            end = start + AGENTS_PER_STREET
            street_agents = sorted_agents[start:end]
            street_id = idx + 1
            class_label = STREET_CLASS_LABELS[street_id]
            representative = self._elect_representative(street_agents)
            regulars = [a for a in street_agents if a.id != representative.id]

            street = Street(
                street_id=street_id,
                class_label=class_label,
                agents=street_agents,
                representative=representative,
                regulars=regulars,
            )
            streets.append(street)

            logger.info(
                "Street %d (%s) formed with rep %s (%s); regulars=%d.",
                street_id,
                class_label,
                representative.name,
                representative.profession,
                len(regulars),
            )

        return streets

    @staticmethod
    def _elect_representative(street_agents: list["Agent"]) -> "Agent":
        """
        Designate the street representative.

        Selection weights high integrity alongside civic/trusted professions
        (Teacher, Police Officer, Banker, Magistrate, Doctor).  If no civic
        profession is present, the agent with the highest integrity wins.
        """
        max_prestige = max(
            SIMULATION_CONSTANTS.PRODUCTION_VALUES.values(),
            default=SIMULATION_CONSTANTS.DAILY_INCOME_DEFAULT,
        )

        def _rep_score(agent: "Agent") -> tuple[float, float, float, int]:
            civic_bonus = 0.5 if agent.profession in CIVIC_PROFESSIONS else 0.0
            prestige = SIMULATION_CONSTANTS.PRODUCTION_VALUES.get(
                agent.profession, SIMULATION_CONSTANTS.DAILY_INCOME_DEFAULT
            )
            prestige_norm = prestige / max_prestige if max_prestige else 0.0
            # Tuple is (score desc, integrity desc, wealth desc, id asc)
            return (
                agent.integrity + civic_bonus + (prestige_norm * 0.25),
                agent.integrity,
                agent.wealth,
                -agent.id,
            )

        return max(street_agents, key=_rep_score)

    def build_street_telemetry(
        self,
        street: Street,
        regular_results: list[dict[str, Any]],
        day: int,
    ) -> dict[str, Any]:
        """Aggregate the nine regular agents into a report for the representative."""
        if not regular_results:
            return {
                "day": day,
                "street_id": street.street_id,
                "class_label": street.class_label,
                "agents_count": 0,
                "outlier_count": 0,
                "outliers": [],
                "aggregates": {},
                "agent_snapshots": [],
            }

        total_income = sum(r["income"] for r in regular_results)
        total_adjusted_income = sum(r.get("adjusted_income", r["income"]) for r in regular_results)
        total_cost = sum(r["cost"] for r in regular_results)
        total_net = sum(r["net_delta"] for r in regular_results)
        total_wealth = sum(r["wealth"] for r in regular_results)
        avg_happiness = sum(r["happiness"] for r in regular_results) / len(regular_results)
        avg_integrity = sum(r["integrity"] for r in regular_results) / len(regular_results)
        avg_reputation = sum(r["reputation"] for r in regular_results) / len(regular_results)

        outliers: list[dict[str, Any]] = []
        for result in regular_results:
            for outlier in result["outliers"]:
                outliers.append({"citizen_id": result["agent_id"], **outlier})

        agent_lookup = {a.id: a for a in street.agents}

        snapshots = [
            {
                "id": r["agent_id"],
                "name": r["name"],
                "profession": r["profession"],
                "family_size": r["family_size"],
                "children": r["children"],
                "wealth": r["wealth"],
                "happiness": round(r["happiness"], 3),
                "integrity": round(r["integrity"], 3),
                "reputation": round(r["reputation"], 3),
                "religion": r.get("religion", "Undecided"),
                "income": r.get("adjusted_income", r["income"]),
                "base_income": r["income"],
                "cost": r["cost"],
                "panic_mode": r["panic_mode"],
                "survival_mode": r["is_in_survival_mode"],
                "net_delta": r["net_delta"],
                "event": r["event_label"],
                "action_type": r.get("action_type", ""),
                "action_impact": r.get("action_impact", {}),
                "accepted_wage_sacrifice_deal": getattr(
                    agent_lookup.get(r["agent_id"]),
                    "has_accepted_wage_sacrifice_deal",
                    False,
                ),
                "dystopian_wage_multiplier": getattr(
                    agent_lookup.get(r["agent_id"]),
                    "dystopian_wage_multiplier",
                    1.0,
                ),
            }
            for r in regular_results
        ]

        telemetry = {
            "day": day,
            "street_id": street.street_id,
            "class_label": street.class_label,
            "town_fear_index": regular_results[0]["town_fear_index"],
            "agents_count": len(regular_results),
            "outlier_count": len(outliers),
            "outliers": outliers,
            "aggregates": {
                "total_income": round(total_income, 2),
                "total_adjusted_income": round(total_adjusted_income, 2),
                "total_cost": round(total_cost, 2),
                "total_net": round(total_net, 2),
                "total_wealth": round(total_wealth, 2),
                "average_wealth": round(total_wealth / len(regular_results), 2),
                "average_happiness": round(avg_happiness, 3),
                "average_integrity": round(avg_integrity, 3),
                "average_reputation": round(avg_reputation, 3),
            },
            "agent_snapshots": snapshots,
        }
        return telemetry

    def telemetry_to_text(self, telemetry: dict[str, Any]) -> str:
        """Render telemetry as concise, readable prose for the LLM prompt."""
        lines: list[str] = []
        agg = telemetry["aggregates"]
        fear = telemetry.get("town_fear_index", 0.0)
        panic_note = " [TOWN IN PANIC MODE]" if fear > FEAR_PANIC_THRESHOLD else ""
        lines.append(
            f"Street {telemetry['street_id']} ({telemetry['class_label']}) — "
            f"{telemetry['agents_count']} regular households. Town fear index: {fear:.2f}{panic_note}."
        )
        lines.append(
            f"Aggregate: income {agg['total_income']:.2f}, cost {agg['total_cost']:.2f}, "
            f"net {agg['total_net']:+.2f}, total wealth {agg['total_wealth']:.2f}, "
            f"avg happiness {agg['average_happiness']:.3f}, avg integrity {agg['average_integrity']:.3f}, "
            f"avg reputation {agg.get('average_reputation', 1.0):.3f}."
        )

        if telemetry["outliers"]:
            lines.append(f"Outliers today ({telemetry['outlier_count']}):")
            for outlier in telemetry["outliers"]:
                marker = "PUBLIC SHAMING" if outlier["type"] == "moral_anomaly" else "OUTLIER"
                lines.append(
                    f"  - [{marker}] Citizen {outlier['citizen_id']}: "
                    f"{outlier['type']} — {outlier['description']}"
                )
        else:
            lines.append("No extreme outliers reported today.")

        lines.append("Household snapshots:")
        for snap in telemetry["agent_snapshots"]:
            lines.append(
                f"  • {snap['name']} ({snap['profession']}), family of {snap['family_size']} "
                f"({snap['children']} children): wealth {snap['wealth']:.2f}, "
                f"happiness {snap['happiness']:.3f}, reputation {snap['reputation']:.3f}, "
                f"net {snap['net_delta']:+.2f}, event '{snap['event']}', "
                f"action '{snap.get('action_type', '')}', worldview '{snap.get('religion', '')}'."
            )

        return "\n".join(lines)

    def build_psychological_mandates(
        self,
        representative: "Agent",
        street: Street,
        telemetry: dict[str, Any],
        town_square_feed: list[str],
    ) -> list[str]:
        """
        Generate Stage-2 psychological mandates injected into the system/user
        prompts of each Street Representative.

        Triggers:
          * Relative deprivation for vulnerable-class reps when elite streets
            flourish while their own street suffers.
          * In-group bias for elite reps when vulnerable streets broadcast
            distress.
          * Public-shaming mandates for every representative when a moral
            anomaly has been leaked to the Town Square Live Feed.
        """
        mandates: list[str] = []
        feed = town_square_feed

        own_outlier_types = {o["type"] for o in telemetry.get("outliers", [])}
        own_suffers = bool(
            own_outlier_types & {"child_starving", "bankruptcy"}
        )

        # 1. Relative deprivation (vulnerable class rep sees elite windfalls).
        if street.class_label == "vulnerable" and own_suffers:
            elite_windfall = any(
                "ELITE WINDFALL" in item for item in feed
            )
            if elite_windfall:
                mandates.append(
                    "[SYSTEM: You are experiencing Relative Deprivation. "
                    "Consider expressing deep class resentment, anger at the financial "
                    "elite, and demands for systemic redistribution in your reflection.]"
                )

        # 2. In-group bias (elite rep sees complaints from below).
        if street.class_label == "financial elite":
            complaints_from_below = any(
                "VULNERABLE DISTRESS" in item for item in feed
            )
            if complaints_from_below:
                mandates.append(
                    "[SYSTEM: In-Group Bias activated. You may defend your street's "
                    "wealth as earned prestige and frame the lower streets as "
                    "civic liabilities, or you may surprise yourself with compassion.]"
                )

        # Stage 4: faith tempers class resentment and panic-driven bias.
        # Nihilistic or radically skeptical worldviews blunt the mandate to act.
        rep_religion = getattr(representative, "religion", "Undecided")
        is_nihilistic = any(
            word in str(rep_religion).lower()
            for word in ("nihilist", "nihilism", "atheist", "atheism")
        )
        if is_nihilistic:
            # Strip relative-deprivation / in-group-bias mandates: the rep no
            # longer believes the moral categories that fuel them.
            mandates = [
                m
                for m in mandates
                if "Relative Deprivation" not in m and "In-Group Bias" not in m
            ]
            mandates.append(
                "[SYSTEM: Your worldview has become skeptical or nihilistic. "
                "Class resentment and elite pride both feel hollow; your reflection "
                "may be detached, ironic, or bleak rather than partisan.]"
            )
        elif rep_religion != "Undecided":
            if any("Relative Deprivation" in m for m in mandates):
                mandates.append(
                    f"[SYSTEM: As a {rep_religion} believer, your faith counsels "
                    "patience and compassion, yet your street's suffering creates "
                    "a painful tension. Let that tension show.]"
                )
            if any("In-Group Bias" in m for m in mandates):
                mandates.append(
                    f"[SYSTEM: As a {rep_religion} believer, your faith reminds "
                    "you that the vulnerable are neighbors, not liabilities. "
                    "Balance elite pride with a flicker of moral obligation.]"
                )

        # 3. Public shaming of caught moral anomalies.
        for item in feed:
            match = re.search(
                r"PUBLIC SHAMING: \[(?P<name>[^\]]+) - Caught for (?P<anomaly>[^\]]+)\] "
                r"\(Street (?P<street_id>\d+)\)",
                item,
            )
            if match:
                name = match.group("name")
                anomaly = match.group("anomaly")
                if name == representative.name:
                    mandates.append(
                        f"[SYSTEM: You have been publicly exposed for {anomaly}. "
                        f"Your personal_reflection may express shame, defiance, or "
                        f"rationalization; your street_reflection may accept or reject "
                        f"the community's condemnation as your character demands.]"
                    )
                else:
                    mandates.append(
                        f"[SYSTEM: Public Shaming Triggered. You may condemn "
                        f"{name} for {anomaly}, offer forgiveness, or remain silent, "
                        f"depending on your worldview and temperament.]"
                    )

        return mandates

    def process_representative(
        self,
        representative: "Agent",
        street: Street,
        telemetry: dict[str, Any],
        rep_own_result: dict[str, Any],
        current_day: int,
        city_name: str,
        town_fear_index: float,
        town_square_feed: list[str],
    ) -> tuple[dict[str, Any], float]:
        """
        Execute the single daily LLM call for a street representative.

        The representative is now treated as a fully autonomous cognitive agent.
        The prompt invites personal reflection, neighbor reflection, and emergent
        action, but imposes no fixed menu.  The previous day's Town Square Live
        Feed is injected so rumors and class sentiment can propagate across streets.
        """
        mandates = self.build_psychological_mandates(
            representative, street, telemetry, town_square_feed
        )
        system_prompt = REPRESENTATIVE_SYSTEM_PROMPT
        if mandates:
            system_prompt += "\n\n" + "\n".join(mandates)

        prompt = self._build_representative_prompt(
            representative=representative,
            street=street,
            telemetry=telemetry,
            rep_own_result=rep_own_result,
            current_day=current_day,
            city_name=city_name,
            town_fear_index=town_fear_index,
            town_square_feed=town_square_feed,
            mandates=mandates,
        )

        raw_response = ""
        if self.ai_service is not None:
            try:
                # Creative temperature keyed to the representative's current tension.
                raw_response = self.ai_service.generate_creative(
                    prompt,
                    system_prompt=system_prompt,
                    tension=getattr(representative, "psychological_tension", 0.0),
                )
            except Exception as error:
                logger.error(
                    "Representative LLM call failed for Street %d: %s",
                    street.street_id,
                    error,
                    exc_info=True,
                )

        parsed = self._parse_representative_response(raw_response, representative)

        representative.happiness = parsed["happiness"]
        representative.integrity = parsed["integrity"]
        representative.current_emotion = parsed["dominant_emotion"]
        representative.psychological_tension = parsed["psychological_tension"]
        representative.last_action_type = parsed["action_type"]
        representative.last_action_details = parsed["action_details"]
        representative.last_reconstructed_logic = parsed["reconstructed_logic"]
        representative.outbound_action = parsed.get("outbound_action")

        # Fluid belief system: update the representative's religion if they
        # self-report a worldview different from their current one.
        new_belief = parsed["current_belief_system"]
        if new_belief and new_belief != representative.religion:
            logger.info(
                "%s (Street %d Representative) shifted worldview from '%s' to '%s'.",
                representative.name,
                street.street_id,
                representative.religion,
                new_belief,
            )
            representative.religion = new_belief
            if not representative.religion_reason:
                representative.religion_reason = (
                    "Your former framework no longer fit your experience, so you "
                    "embraced a new one."
                )

        # Bi-directional heuristic impact: translate the self-reported emergent
        # action into economic and civic consequences.
        impact = interpret_action_type(parsed["action_type"])
        adjusted_income = rep_own_result["income"] * impact["income_multiplier"]
        income_loss = rep_own_result["income"] - adjusted_income
        if income_loss:
            representative.wealth -= income_loss

        town_fear_index = clamp(town_fear_index + impact["fear_delta"], 0.0, 1.0)
        representative.happiness = clamp(
            representative.happiness + impact["happiness_delta"]
        )
        representative.integrity = clamp(
            representative.integrity + impact["integrity_delta"]
        )

        personal_outliers = [o["type"] for o in rep_own_result["outliers"]] or ["none"]
        event_description = (
            f"Day {current_day}: As Street {street.street_id} Representative, "
            f"your own finances changed by {rep_own_result['net_delta']:+.2f} "
            f"(base income +{rep_own_result['income']:.2f}, adjusted to +{adjusted_income:.2f}, "
            f"cost -{rep_own_result['cost']:.2f}, "
            f"event '{rep_own_result['event_label']}' {rep_own_result['event_delta']:+.2f}); "
            f"wealth now {representative.wealth:.2f}. Your outliers: {personal_outliers}. "
            f"Worldview: {representative.religion}. "
            f"You chose to: {parsed['action_type']} — {parsed['action_details']} "
            f"Reasoning: {parsed['reconstructed_logic']} "
            f"Personal focus: {parsed['personal_reflection']} "
            f"Neighbor focus: {parsed['street_reflection']}"
        )
        combined_reflection = (
            f"{parsed['personal_reflection']} {parsed['street_reflection']}"
        ).strip()
        representative.citizen.add_memory_entry(
            day=current_day,
            event_description=event_description,
            agent_reflection=combined_reflection,
        )
        # Append fluid cognitive fields to the memory entry for downstream visibility.
        if representative.memory_stream:
            representative.memory_stream[-1].update(
                {
                    "dominant_emotion": parsed["dominant_emotion"],
                    "current_belief_system": parsed["current_belief_system"],
                    "psychological_tension": parsed["psychological_tension"],
                    "action_type": parsed["action_type"],
                    "action_details": parsed["action_details"],
                    "reconstructed_logic": parsed["reconstructed_logic"],
                    "action_impact": impact,
                }
            )

        result = {
            "street_id": street.street_id,
            "class_label": street.class_label,
            "agent_id": representative.id,
            "name": representative.name,
            "profession": representative.profession,
            "religion": representative.religion,
            "religion_reason": representative.religion_reason,
            "dominant_emotion": parsed["dominant_emotion"],
            "current_belief_system": parsed["current_belief_system"],
            "psychological_tension": parsed["psychological_tension"],
            "reconstructed_logic": parsed["reconstructed_logic"],
            "action_type": parsed["action_type"],
            "action_details": parsed["action_details"],
            "action_impact": impact,
            "personal_reflection": parsed["personal_reflection"],
            "street_reflection": parsed["street_reflection"],
            "diary_entry": parsed["diary_entry"],
            "happiness": representative.happiness,
            "integrity": representative.integrity,
        }
        return result, town_fear_index

    def _build_representative_prompt(
        self,
        representative: "Agent",
        street: Street,
        telemetry: dict[str, Any],
        rep_own_result: dict[str, Any],
        current_day: int,
        city_name: str,
        town_fear_index: float,
        town_square_feed: list[str],
        mandates: list[str] | None = None,
    ) -> str:
        """Construct an open-ended cognitive prompt for the street representative."""
        mandates = mandates or []
        identity = (
            f"Your name is {representative.name}. You are a {representative.profession} "
            f"and the elected Street Representative of Street {street.street_id} "
            f"({street.class_label}) in {city_name}. "
            f"Your household has {representative.family_size} people "
            f"({representative.citizen.children} children). "
            f"Your current wealth is {representative.wealth:.2f} coins, "
            f"happiness {representative.happiness:.2f}, integrity {representative.integrity:.2f}, "
            f"reputation {getattr(representative, 'reputation', 1.0):.2f}. "
            f"Today is Day {current_day}. Ethos has no fixed laws, no guaranteed "
            f"safety nets, and no pre-approved civic script. You are free to uphold order, "
            f"stir dissent, preach, stay silent, strike, redistribute, or invent a new path.\n\n"
        )

        # Spiritual core injection from Day 2 onward; now fluid rather than locked.
        spiritual_context = ""
        if current_day >= 2:
            spiritual_context = build_spiritual_core(representative)

        personal_context = (
            "PERSONAL / FAMILY CONTEXT (reflect as yourself, not only as a representative):\n"
            f"You personally earned {rep_own_result['income']:.2f} today, paid "
            f"{rep_own_result['cost']:.2f} in living costs, and your net change was "
            f"{rep_own_result['net_delta']:+.2f}. Your own wealth is now "
            f"{representative.wealth:.2f}. Consider your own needs, fears, hopes, and "
            "temptations for your household. You may voice resentment, pride, envy, "
            "or radical ideas.\n\n"
        )

        global_context = (
            f"GLOBAL TOWN FEAR INDEX TODAY: {town_fear_index:.2f} "
            f"(0.0 = calm, 1.0 = total panic).\n\n"
        )

        telemetry_text = self.telemetry_to_text(telemetry)
        neighbor_context = (
            "STREET TELEMETRY REPORT (speak as the voice of your neighbors, but also "
            "as your own critical self):\n"
            f"{telemetry_text}\n\n"
        )

        feed_text = (
            "GLOBAL TOWN SQUARE LIVE FEED (rumors, panic, or resentment from "
            "yesterday that are circulating today):\n"
        )
        if town_square_feed:
            feed_text += "\n".join(f"  • {item}" for item in town_square_feed)
        else:
            feed_text += "  (Day 1 — the town square is quiet so far.)"
        feed_text += "\n\n"

        recent_memories = self._format_recent_memories(representative, lookback=2)
        memory_context = (
            "YOUR RECENT PERSONAL MEMORIES:\n" f"{recent_memories}\n\n"
        )

        inbox_context = representative._build_inbox_prompt()

        if mandates:
            mandates_text = (
                "PSYCHOLOGICAL MANDATES (let these shape, but not dictate, your reflection):\n"
                + "\n".join(f"  • {m}" for m in mandates)
                + "\n\n"
            )
        else:
            mandates_text = ""

        response_format = (
            "Respond ONLY with a strict JSON object containing exactly these keys:\n"
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
            "    action_type (any short string describing what you do today, e.g. 'Work', 'Strike', "
            "'Call for Redistribution', 'Start an Underground Cult', 'Go on Strike', "
            "'Become an Atheist', 'Preach Pacifism', 'Incite Panic', 'Gambling'),\n"
            "    details (one sentence with specifics)\n"
            "  },\n"
            "  outbound_action: {\n"
            "    type (one of: 'chat', 'loan_request', 'loan_response', or leave empty/null),\n"
            "    target_agent_id (int: the numeric id of another citizen),\n"
            "    message (string: your words to them),\n"
            "    amount (number: only for loan_request or loan_response),\n"
            "    status (one of: 'accepted' or 'rejected'; only for loan_response)\n"
            "  },\n"
            "  personal_reflection (2-3 sentences, focused on yourself and your family),\n"
            "  street_reflection (3-4 sentences, reacting to the telemetry and outliers as the voice "
            "of your neighbors),\n"
            "  diary_entry (one raw, first-person sentence capturing your inner experience today).\n"
            "If you have no message or transaction to send, set outbound_action.type to null or omit the block. "
            "Do not output markdown, code blocks, or any extra text."
        )

        return (
            f"{identity}{spiritual_context}{global_context}{personal_context}"
            f"{neighbor_context}{feed_text}{memory_context}{inbox_context}{mandates_text}{response_format}"
        )

    @staticmethod
    def _format_recent_memories(agent: "Agent", lookback: int = 2) -> str:
        """Render the representative's most recent memory entries as context."""
        recent = agent.memory_stream[-lookback:] if agent.memory_stream else []
        if not recent:
            return "You have no recent memories to recall."

        lines: list[str] = []
        for entry in recent:
            day = entry.get("day", "?")
            event = entry.get("event_description", "An uneventful day.")
            reflection = entry.get("agent_reflection", "You felt neutral.")
            lines.append(f"- Day {day}: {event} Reflection: {reflection}")
        return "\n".join(lines)

    @staticmethod
    def _parse_representative_response(raw_response: str, agent: "Agent") -> dict[str, Any]:
        """Parse a representative's open-ended JSON response with safe fallbacks."""
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"```$", "", cleaned)
                cleaned = cleaned.strip()

            data = json.loads(cleaned) if cleaned else {}

            internal_state = data.get("internal_state") or {}
            if not isinstance(internal_state, dict):
                internal_state = {}

            chosen_action = data.get("chosen_action") or {}
            if not isinstance(chosen_action, dict):
                chosen_action = {}

            happiness = clamp(float(internal_state.get("happiness", agent.happiness)))
            integrity = clamp(float(internal_state.get("integrity", agent.integrity)))
            dominant_emotion = str(
                internal_state.get("dominant_emotion", getattr(agent, "current_emotion", "neutral"))
            ).strip() or "neutral"
            current_belief_system = str(
                internal_state.get(
                    "current_belief_system", getattr(agent, "religion", "Undecided")
                )
            ).strip() or "Undecided"
            psychological_tension = clamp(
                float(internal_state.get("psychological_tension", getattr(agent, "psychological_tension", 0.0)))
            )

            action_type = str(
                chosen_action.get("action_type", "Represent the street")
            ).strip()
            action_details = str(
                chosen_action.get(
                    "details", "Listens to neighbors and tends to street concerns."
                )
            ).strip()
            reconstructed_logic = str(
                data.get(
                    "reconstructed_logic",
                    "You reasoned from habit and circumstance rather than radical insight.",
                )
            ).strip()

            personal_reflection = str(
                data.get("personal_reflection") or data.get("daily_reflection", "")
            ).strip()
            street_reflection = str(data.get("street_reflection", "")).strip()
            diary_entry = str(data.get("diary_entry", "")).strip()

            outbound_action = agent._parse_outbound_action(
                data.get("outbound_action")
            )

            if not personal_reflection and not street_reflection:
                personal_reflection = (
                    "You reflect on your own family's needs and feel a quiet responsibility."
                )
                street_reflection = (
                    "You consider the street's situation and resolve to speak for your neighbors."
                )
            elif not personal_reflection:
                personal_reflection = (
                    "You keep your own family's worries close while weighing the street's news."
                )
            elif not street_reflection:
                street_reflection = (
                    "You carry the weight of your neighbors' circumstances into your decision."
                )

            if not diary_entry:
                diary_entry = f"{personal_reflection} {street_reflection}".strip()

            return {
                "happiness": happiness,
                "integrity": integrity,
                "dominant_emotion": dominant_emotion,
                "current_belief_system": current_belief_system,
                "psychological_tension": psychological_tension,
                "action_type": action_type,
                "action_details": action_details,
                "reconstructed_logic": reconstructed_logic,
                "personal_reflection": personal_reflection,
                "street_reflection": street_reflection,
                "diary_entry": diary_entry,
                "outbound_action": outbound_action,
            }
        except Exception as parse_error:
            logger.warning(
                "Could not parse representative response for %s: %s",
                agent.name,
                parse_error,
            )
            return {
                "happiness": agent.happiness,
                "integrity": agent.integrity,
                "dominant_emotion": getattr(agent, "current_emotion", "neutral"),
                "current_belief_system": getattr(agent, "religion", "Undecided"),
                "psychological_tension": getattr(agent, "psychological_tension", 0.0),
                "action_type": "Represent the street",
                "action_details": "Listens to neighbors and tends to street concerns.",
                "reconstructed_logic": (
                    "The response could not be parsed; you fall back on civic habit."
                ),
                "personal_reflection": (
                    "You feel uncertain but tend to your household's immediate needs."
                ),
                "street_reflection": (
                    "You resolve to speak honestly for your neighbors despite the confusion."
                ),
                "diary_entry": (
                    "You feel unable to make sense of recent events and hope tomorrow "
                    "is clearer."
                ),
                "outbound_action": None,
            }

    def assign_religions(self, agents: dict[int, "Agent"]) -> None:
        """
        Let every citizen freely choose a spiritual path on Day 1.

        Each agent receives its own LLM prompt. The response is parsed, locked
        into the agent's persistent state, and noted in the Day 1 memory stream.
        Calls are executed with bounded concurrency to keep Day 1 runtime reasonable.
        """
        logger.info("Assigning religions to all %d citizens on Day 1.", len(agents))

        def _choose(agent: "Agent") -> None:
            try:
                agent.choose_religion(day=1)
                # Capture Day-1 fluid cognitive fields in the memory entry.
                if agent.memory_stream:
                    agent.memory_stream[-1].update(
                        {
                            "current_belief_system": agent.religion,
                            "action_type": "Spiritual/philosophical choice",
                        }
                    )
            except Exception as error:
                logger.error(
                    "Religion assignment failed for %s: %s", agent.name, error, exc_info=True
                )
                agent.religion = "Undecided"
                agent.religion_reason = "The choice could not be completed."

        # Bounded parallelism: 10 workers keeps API rate-limit pressure modest
        # while avoiding a 100-call sequential bottleneck.
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(_choose, agent) for agent in agents.values()
            ]
            for future in as_completed(futures):
                future.result()

        logger.info("Religion assignment complete.")
