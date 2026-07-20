"""
Economic and behavioural baseline constants for ETHOS.

These values are intentionally immutable defaults. Tune them via environment
variables or settings only when the simulation design changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConstants:
    """Immutable container for simulation baseline values."""

    # Citizen affect defaults (continuous 0.0-1.0)
    INITIAL_HAPPINESS: float = 1.0
    INITIAL_INTEGRITY: float = 1.0

    # LLM / API defaults
    DEFAULT_TEMPERATURE: float = 0.7
    RESPONSE_FORMAT: str = "json_object"

    # Economy
    DAILY_INCOME_DEFAULT: float = 50.0
    DEFAULT_TAX_RATE: float = 0.0
    BASELINE_PRICE_LEVEL: float = 1.0

    # Profession-specific daily production values (coins)
    PRODUCTION_VALUES: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # dataclasses with mutable defaults require late initialisation
        object.__setattr__(
            self,
            "PRODUCTION_VALUES",
            {
                "Police Officer": 100.0,
                "Carpenter": 85.0,
                "Farmer": 60.0,
                "Merchant": 90.0,
                "Teacher": 80.0,
                "Landlord": 150.0,
                "Doctor": 110.0,
                "Banker": 200.0,
                "Magistrate": 180.0,
                "Goldsmith": 160.0,
                "Jeweler": 155.0,
                "Architect": 145.0,
                "Mercer": 140.0,
                "Tax Collector": 135.0,
                "Innkeeper": 130.0,
                "Vintner": 125.0,
                "Silversmith": 120.0,
                "Alchemist": 115.0,
                "Pharmacist": 112.0,
                "Blacksmith": 95.0,
                "Butcher": 92.0,
                "Shipwright": 90.0,
                "Armourer": 88.0,
                "Navigator": 87.0,
                "Cartographer": 86.0,
                "Surveyor": 85.0,
                "Astrologer": 84.0,
                "Engraver": 83.0,
                "Watchmaker": 82.0,
                "Clockmaker": 81.0,
                "Chef": 80.0,
                "Sculptor": 78.0,
                "Fletcher": 77.0,
                "Musician": 76.0,
                "Brewer": 75.0,
                "Bookbinder": 74.0,
                "Calligrapher": 73.0,
                "Historian": 72.0,
                "Miller": 71.0,
                "Saddler": 70.0,
                "Stablemaster": 69.0,
                "Falconer": 68.0,
                "Ironmonger": 67.0,
                "Soapmaker": 66.0,
                "Leather Worker": 65.0,
                "Painter": 64.0,
                "Shoemaker": 63.0,
                "Potter": 62.0,
                "Baker": 61.0,
                "Barber": 60.0,
                "Netmaker": 58.0,
                "Courier": 57.0,
                "Wheelwright": 56.0,
                "Glassblower": 55.0,
                "Roofer": 54.0,
                "Coppersmith": 53.0,
                "Mason": 52.0,
                "Locksmith": 51.0,
                "Weaver": 50.0,
                "Tanner": 49.0,
                "Cooper": 48.0,
                "Dyer": 47.0,
                "Gamekeeper": 46.0,
                "Fishmonger": 45.0,
                "Scholar": 44.0,
                "Ropemaker": 43.0,
                "Jailer": 42.0,
                "Bowyer": 41.0,
                "Bricklayer": 40.0,
                "Grocer": 39.0,
                "Chandler Seller": 38.0,
                "Chandler": 37.0,
                "Fisherman": 36.0,
                "Nightwatchman": 35.0,
                "Pottery Seller": 34.0,
                "Well Digger": 33.0,
                "Millwright": 32.0,
                "Bailiff": 31.0,
                "Scribe": 30.0,
                "Miner": 28.0,
                "Quarryman": 27.0,
                "Logger": 26.0,
                "Charcoal Burner": 25.0,
                "Chimney Sweep": 24.0,
                "Sailor": 23.0,
                "Cart Driver": 22.0,
                "Basketmaker": 21.0,
                "Shepherd": 20.0,
                "Philosopher": 18.0,
                "Rat Catcher": 15.0,
            },
        )


SIMULATION_CONSTANTS = SimulationConstants()
