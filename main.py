"""
ETHOS — Multi-Agent Socio-Economic Simulation

System entry point. Configures logging and runs the simulation.
Database persistence is handled internally by SimulationEngine.
"""

from config.settings import get_settings
from services.simulation import SimulationEngine
from utils.logger import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level)

    engine = SimulationEngine()
    engine.run()


if __name__ == "__main__":
    main()
