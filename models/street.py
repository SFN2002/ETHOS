"""Socio-economic street abstraction for ETHOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.agent import Agent


@dataclass
class Street:
    """A socio-economic zone containing exactly ten citizen agents."""

    street_id: int
    class_label: str
    agents: list[Agent]
    representative: Agent
    regulars: list[Agent]
    daily_telemetry: list[dict[str, Any]] = field(default_factory=list)

    @property
    def population(self) -> int:
        return len(self.agents)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable street schema compatible with downstream analytics."""
        return {
            "street_id": self.street_id,
            "class_label": self.class_label,
            "population": self.population,
            "representative_id": self.representative.id,
            "representative_name": self.representative.name,
            "representative_profession": self.representative.profession,
            "representative_religion": getattr(self.representative, "religion", "Undecided"),
            "representative_religion_reason": getattr(self.representative, "religion_reason", ""),
            "representative_emotion": getattr(self.representative, "current_emotion", "neutral"),
            "representative_tension": getattr(self.representative, "psychological_tension", 0.0),
            "representative_last_action": getattr(self.representative, "last_action_type", ""),
            "representative_sacrificed": getattr(
                self.representative, "has_accepted_wage_sacrifice_deal", False
            ),
            "representative_sacrificed_member": getattr(
                self.representative, "sacrificed_family_member", "none"
            ),
            "agent_ids": [agent.id for agent in self.agents],
            "regular_agent_ids": [agent.id for agent in self.regulars],
            "agent_cognitive_states": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "profession": agent.profession,
                    "religion": getattr(agent, "religion", "Undecided"),
                    "religion_reason": getattr(agent, "religion_reason", ""),
                    "dominant_emotion": getattr(agent, "current_emotion", "neutral"),
                    "psychological_tension": getattr(agent, "psychological_tension", 0.0),
                    "last_action_type": getattr(agent, "last_action_type", ""),
                    "last_action_details": getattr(agent, "last_action_details", ""),
                    "accepted_wage_sacrifice_deal": getattr(
                        agent, "has_accepted_wage_sacrifice_deal", False
                    ),
                    "sacrificed_family_member": getattr(
                        agent, "sacrificed_family_member", "none"
                    ),
                    "dystopian_wage_multiplier": getattr(
                        agent, "dystopian_wage_multiplier", 1.0
                    ),
                }
                for agent in self.agents
            ],
            "daily_telemetry": self.daily_telemetry,
        }
