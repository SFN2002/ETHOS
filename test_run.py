import os
os.environ["DEEPSEEK_API_KEY"] = ""

from services.simulation import SimulationEngine

engine = SimulationEngine()
# Seed a drought rumor to accelerate fear and demonstrate narrative event impact.
engine.city.seed_narrative_event("drought_rumor")

summary = engine.run(days=10)
print("Summary keys:", list(summary.keys()))
print("Final fear index:", summary["final_metrics"]["town_fear_index"])
print("Fear history:", summary["fear_history"])
print("Public shaming log entries:", len(summary["public_shaming_log"]))
# Find an agent whose integrity collapsed under fear
for a in summary["agent_summaries"]:
    if a["integrity"] < 0.5 and a["reputation"] < 0.2:
        print("Moral collapse pariah:", a["name"], "integrity", a["integrity"], "reputation", a["reputation"])
        break
