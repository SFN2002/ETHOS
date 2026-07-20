"""
Pydantic V2 schema for a Ethos citizen.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from config.constants import SIMULATION_CONSTANTS


class Citizen(BaseModel):
    """
    Data model describing a citizen and their mutable simulation state.

    The ``memory_stream`` is the citizen's generative memory: a chronologically
    ordered record of daily events, family snapshots, and internal reflections.
    It is mutated by the :class:`~models.agent.Agent` wrapper during simulation.

    Attributes:
        id: Unique citizen identifier (1-100).
        name: Citizen first name.
        profession: Citizen occupation.
        wealth: Current economic endowment (non-negative).
        status: Marital status.
        sons: Number of male children.
        daughters: Number of female children.
        happiness: Continuous affect state in [0.0, 1.0].
        integrity: Continuous moral-trust state in [0.0, 1.0].
        memory_stream: Chronological stream of daily memory entries.
    """

    id: int = Field(..., ge=1, le=100, description="Unique citizen identifier.")
    name: str = Field(..., min_length=1, description="Citizen first name.")
    profession: str = Field(..., min_length=1, description="Occupation.")
    wealth: float = Field(..., ge=0.0, description="Initial economic endowment.")
    status: Literal["married", "single", "divorced", "widowed"] = Field(
        ..., description="Marital status."
    )
    sons: int = Field(default=0, ge=0, description="Number of male children.")
    daughters: int = Field(
        default=0, ge=0, description="Number of female children."
    )
    happiness: float = Field(
        default=SIMULATION_CONSTANTS.INITIAL_HAPPINESS,
        ge=0.0,
        le=1.0,
        description="Continuous affect state.",
    )
    integrity: float = Field(
        default=SIMULATION_CONSTANTS.INITIAL_INTEGRITY,
        ge=0.0,
        le=1.0,
        description="Continuous moral-trust state.",
    )
    memory_stream: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological stream of daily memory entries.",
    )

    @field_validator("happiness", "integrity", mode="before")
    @classmethod
    def _coerce_float(cls, value: float) -> float:
        """Ensure happiness and integrity values are stored as floats.

        Args:
            value: The incoming numeric value.

        Returns:
            The value coerced to a float.
        """
        return float(value)

    @property
    def family_size(self) -> int:
        """
        Total household size.

        Single citizens are assumed to live alone; all other statuses include
        the citizen plus dependent children.

        Returns:
            Integer household size.
        """
        if self.status == "single":
            return 1
        return 1 + self.sons + self.daughters

    @property
    def children(self) -> int:
        """Return the total number of children.

        Returns:
            Sum of sons and daughters.
        """
        return self.sons + self.daughters

    def _build_family_status_snapshot(self) -> str:
        """Return a concise, human-readable family snapshot.

        Returns:
            String describing marital status and children.
        """
        child_phrase = "no children"
        if self.children == 1:
            child_phrase = "1 child"
        elif self.children > 1:
            child_phrase = f"{self.children} children"
            if self.sons > 0 and self.daughters > 0:
                child_phrase = f"{self.sons} son{'s' if self.sons > 1 else ''} and {self.daughters} daughter{'s' if self.daughters > 1 else ''}"

        if self.status == "single":
            return f"Single, living alone with {child_phrase}."
        return f"{self.status.capitalize()}, with {child_phrase}."

    def add_memory_entry(
        self,
        day: int,
        event_description: str,
        agent_reflection: str,
    ) -> dict[str, Any]:
        """
        Append a structured memory entry to the citizen's memory stream.

        Args:
            day: Simulation day associated with the memory.
            event_description: Objective description of the day's event.
            agent_reflection: Subjective reflection on the event.

        Returns:
            The dictionary entry that was appended to the memory stream.
        """
        entry = {
            "day": int(day),
            "event_description": str(event_description).strip(),
            "family_status_snapshot": self._build_family_status_snapshot(),
            "agent_reflection": str(agent_reflection).strip(),
        }
        self.memory_stream.append(entry)
        return entry

    def model_dump_public(self) -> dict:
        """Return a JSON-serialisable view of the citizen.

        Returns:
            Dictionary representation of the citizen model.
        """
        return self.model_dump(mode="json")
