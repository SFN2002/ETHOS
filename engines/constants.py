"""
Simulation constants extracted from the legacy monolithic city controller.

These values are intentionally untouched: moving them here is purely a
structural refactor to support the engines/ architecture.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Socio-economic zoning constants
# ---------------------------------------------------------------------------
EXPECTED_POPULATION: int = 100
STREET_COUNT: int = 10
AGENTS_PER_STREET: int = 10

COST_PER_CAPITA: float = 8.0
COST_PER_CHILD: float = 2.5

# Deterministic event economy
EVENT_WINDFALL_PROBABILITY: float = 0.06
EVENT_MISFORTUNE_PROBABILITY: float = 0.10  # cumulative after windfall
WINDFALL_MULTIPLIER_MIN: float = 1.0
WINDFALL_MULTIPLIER_MAX: float = 4.0
MISFORTUNE_MULTIPLIER_MIN: float = 0.5
MISFORTUNE_MULTIPLIER_MAX: float = 2.5

OUTLIER_WINDFALL_THRESHOLD: float = 100.0
OUTLIER_MAJOR_LOSS_THRESHOLD: float = -75.0

# Affective drift applied by the fast-math engine
HAPPINESS_NET_POSITIVE_BUMP: float = 0.01
HAPPINESS_NET_NEGATIVE_BUMP: float = -0.02
HAPPINESS_WINDFALL_BUMP: float = 0.04
HAPPINESS_MISFORTUNE_BUMP: float = -0.05
HAPPINESS_BANKRUPTCY_BUMP: float = -0.15
HAPPINESS_STARVATION_BUMP: float = -0.20

INTEGRITY_WINDFALL_BUMP: float = 0.01
INTEGRITY_BANKRUPTCY_BUMP: float = -0.03
INTEGRITY_STARVATION_BUMP: float = -0.05
INTEGRITY_MORAL_ANOMALY_BUMP: float = -0.20
INTEGRITY_HIDDEN_GUILT_BUMP: float = -0.05

# Moral anomaly / public shaming constants
MORAL_ANOMALY_INTEGRITY_THRESHOLD: float = 0.50
MORAL_ANOMALY_PROBABILITY: float = 0.30
MORAL_ANOMALY_CATCH_PROBABILITY: float = 0.80
PARIAH_REPUTATION: float = 0.10
PARIAH_HAPPINESS: float = 0.10
ELITE_NET_WINDFALL_THRESHOLD: float = 500.0

# Stage 3: town fear / panic buying / moral collapse
FEAR_DECAY: float = 0.08
FEAR_VULNERABLE_DISTRESS_BUMP: float = 0.10
FEAR_MORAL_ANOMALY_BUMP: float = 0.14
FEAR_NARRATIVE_BUMP: float = 0.22
FEAR_PANIC_THRESHOLD: float = 0.50
FEAR_MORAL_COLLAPSE_THRESHOLD: float = 0.60
SURVIVAL_INTEGRITY_PENALTY: float = -0.15
PHYSIOLOGICAL_COST_SHARE: float = 0.70
SOCIAL_ESTEEM_COST_SHARE: float = 0.30

# Stage 4: religious autonomy / spiritual core
RELIGIOUS_INTEGRITY_RESILIENCE: float = 0.40
RELIGIOUS_PANIC_DISCOUNT: float = 0.35

CIVIC_PROFESSIONS: set[str] = {
    "Teacher",
    "Police Officer",
    "Banker",
    "Magistrate",
    "Doctor",
}

REPRESENTATIVE_SYSTEM_PROMPT: str = (
    "You are the elected Street Representative of a socio-economic zone in "
    "Ethos, but first and foremost you are a free, autonomous human being. "
    "You have your own family, fears, desires, and worldview. You also speak for "
    "your neighbors, yet you are not bound to any civic script. You may defend "
    "the status quo, demand redistribution, preach, renounce your faith, call for "
    "a strike, incite panic, calm the crowd, or take any other action that emerges "
    "from your circumstances and temperament. Respond ONLY with a valid JSON object "
    "matching the requested schema. Do not include markdown, code blocks, or any extra text."
)

STREET_CLASS_LABELS: dict[int, str] = {
    1: "vulnerable",
    2: "vulnerable",
    3: "working class",
    4: "working class",
    5: "middle class",
    6: "middle class",
    7: "upper-middle class",
    8: "upper-middle class",
    9: "financial elite",
    10: "financial elite",
}
