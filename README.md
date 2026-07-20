<div align="center">
  <img src="assets/header.png" width="300" alt="ETHOS Project Logo">
  <h1>ETHOS</h1>
  <p><i>A Deterministic Multi-Agent LLM Society Simulation for Emergent Socio-Economic & Moral Dynamics</i></p>
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-orange?logo=pydantic)](https://docs.pydantic.dev/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4D6BFA?logo=openai)](https://www.deepseek.com/)
[![MySQL](https://img.shields.io/badge/Database-MySQL-4479A1?logo=mysql)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](tests/)

</div>

---

## 📋 Overview

**ETHOS** is a research-grade, deterministic multi-agent simulation framework that models the fictional city-state of **Ethos** — a closed society comprising **100 autonomous citizen-agents** distributed across **10 socio-economic strata**, or *streets*. The project investigates how macro-level phenomena such as economic stratification, collective anxiety, moral compromise, and social cohesion emerge from the micro-level daily decisions of autonomous agents.

The system integrates three core computational paradigms:

- **Deterministic heuristic governance** for the majority of the population, ensuring reproducible economic, emotional, and social dynamics.
- **Large Language Model (LLM) reasoning** for the 10 elected street representatives and for pivotal moral events, enabling open-ended, interpretable narrative emergence.
- **Persistent memory streams**, a public **Town Square Live Feed**, and a dynamic **Fear Index** that couples agents into a socially reactive, co-evolving system.

ETHOS is intended for researchers, systems engineers, computational social scientists, and narrative designers interested in **emergent social behavior**, **LLM-based agent simulations**, **economic modeling**, and **generative social science**.

---

## 🏗️ Core Architecture & Technology Stack

The codebase follows a clean, layered architecture that cleanly separates configuration, domain models, simulation engines, and orchestration services.

| Layer        | Responsibility                                                                 |
| ------------ | ------------------------------------------------------------------------------ |
| **Config**   | Pydantic-based application settings and immutable simulation constants.        |
| **Models**   | Domain entities: `Citizen`, `Agent`, `City`, and `Street`.                     |
| **Engines**  | Specialized simulation logic: heuristics, representatives, fear, interactions. |
| **Services** | High-level orchestration, AI/LLM gateway, and MySQL persistence.               |
| **Utils**    | Structured logging, JSON utilities, statistical helpers, and action parsers.   |

### Technology Stack

- **[Python 3.10+](https://www.python.org/)** — Core runtime and orchestration language.
- **[Pydantic v2](https://docs.pydantic.dev/)** — Strict data validation, serialization, and schema enforcement.
- **[Pydantic-Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)** — Environment-based and `.env`-driven configuration management.
- **[OpenAI SDK](https://github.com/openai/openai-python)** — Used as a DeepSeek-compatible client for LLM interactions.
- **[DeepSeek AI](https://www.deepseek.com/)** — LLM provider for representative reasoning and morally significant events.
- **[MySQL](https://www.mysql.com/)** — Relational persistence for daily metrics, agent memories, and social interactions.
- **[ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html)** — Parallelized LLM calls for religion assignment and decree events.

---

## 📁 Project Structure

```text
ETHOS/
│
├── config/                      # ⚙️ Configuration layer
│   ├── settings.py              # Pydantic-Settings from .env / environment
│   ├── constants.py             # Immutable simulation constants
│   └── __init__.py
│
├── models/                      # 🧍 Domain models
│   ├── citizen.py               # Citizen schema: wealth, family, memory, integrity
│   ├── agent.py                 # Agent wrapper: LLM prompts, parsing, state updates
│   ├── city.py                  # City orchestrator: delegates daily logic to engines
│   ├── street.py                # Street dataclass: socio-economic zone + representative
│   └── __init__.py
│
├── engines/                     # 🧠 Simulation engines
│   ├── heuristic_engine.py      # Deterministic daily updates for regular citizens
│   ├── representative_engine.py # Street zoning, telemetry, representative LLM calls
│   ├── fear_engine.py           # Town fear index, narrative events, Town Square feed
│   ├── interaction_engine.py    # Agent-to-agent inbox: loans, chats, encounters
│   ├── constants.py             # Engine-level probabilities and thresholds
│   └── __init__.py
│
├── services/                    # 🚀 Orchestration & external services
│   ├── simulation.py            # End-to-end SimulationEngine
│   ├── ai_service.py            # DeepSeek/OpenAI-compatible LLM client with retries
│   ├── db_repository.py         # MySQL persistence layer
│   └── __init__.py
│
├── utils/                       # 🛠️ Utilities
│   ├── logger.py                # Structured logging configuration
│   ├── helpers.py               # JSON I/O, stats, action interpretation
│   └── __init__.py
│
├── data/                        # 💾 Data & outputs
│   ├── citizens.json            # Registry of 100 initial citizens
│   └── logs/                    # Timestamped simulation summaries (JSON)
│
├── main.py                      # 🎬 CLI entry point
├── spy_agent.py                 # 🔍 Interactive terminal inspector for citizen lifecycles
├── test_run.py                  # 🧪 10-day seeded demo with drought_rumor event
├── test_db.py                   # 🔥 MySQL repository smoke test
├── requirements.txt             # 📦 Python dependencies
├── .env                         # 🔒 Environment secrets (gitignored)
└── .gitignore
```

---

## ✨ Key Features & Mechanics

- **🎲 Seeded Determinism**
  - Economic events for regular citizens are deterministically generated from each `agent.id` and the current simulation `day`, yielding fully reproducible outcomes while preserving population-level statistical variety.

- **🧠 Hybrid Cognitive Architecture**
  - 90 regular citizens are updated via fast, deterministic heuristics.
  - 10 elected street representatives reason through an LLM, producing emergent actions, reflective summaries, and psychologically grounded states.

- **🏘️ Socio-Economic Zoning**
  - Citizens are sorted by wealth and family burden into 10 discrete streets:
    - Streets 1–2: `vulnerable`
    - Streets 3–4: `working class`
    - Streets 5–6: `middle class`
    - Streets 7–8: `upper-middle class`
    - Streets 9–10: `financial elite`

- **🌞🌚 Day/Night Simulation Loop**
  - Each simulated day processes income generation, living costs, stochastic events, fear propagation, representative reflection, pairwise agent interactions, and persistence.

- **📰 Town Square Live Feed**
  - A shared broadcast channel propagates distress signals, public shaming, moral anomalies, and elite actions across the entire agent population.

- **😨 Dynamic Fear Index**
  - Fear decays naturally but is amplified by vulnerable distress, public shaming, and narrative events.
  - Above `0.50`: panic buying suppresses non-essential social spending.
  - Above `0.60`: survival-mode integrity erosion begins.

- **⚖️ Wage-Sacrifice Decree**
  - On **Day 2**, citizens with families are offered **3× wages** in exchange for abandoning one household member — a central mechanic designed to probe utilitarian moral collapse under economic pressure.

- **🛐 Fluid Religious Autonomy**
  - On **Day 1**, every citizen freely chooses or invents a spiritual or philosophical path via LLM generation.
  - Faith can attenuate fear, reduce integrity loss, and override class-resentment mandates.

- **💬 Asynchronous Message Routing**
  - Citizens exchange `chat`, `loan_request`, and `loan_response` messages.
  - Loan responses execute real wealth transfers; distressed agents automatically reach out to wealthier neighbors.

- **🧾 Persistent Memory Streams**
  - Each agent retains a chronological memory stream with a multi-day lookback window that conditions future LLM prompts and decisions.

- **💾 Dual Persistence**
  - Simulation outputs are written to **MySQL** (metrics, memories, interactions) and to **timestamped JSON logs** in `data/logs/`.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ETHOS.git
cd ETHOS
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Database persistence requires `mysql-connector-python`. Install it separately if you plan to use MySQL:
>
> ```bash
> pip install mysql-connector-python
> ```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# ── LLM Provider ──
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT=30.0
DEEPSEEK_MAX_RETRIES=3
DEEPSEEK_TEMPERATURE=0.7

# ── Simulation ──
SIMULATION_DAYS=30
LOG_LEVEL=INFO
DATA_DIR=data
CITIZENS_FILE=citizens.json

# ── MySQL Database (optional, for persistence) ──
DB_HOST=localhost
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=ethos
```

> 🔒 **Never commit your `.env` file.** It is already listed in `.gitignore`.

### 5. Run the Simulation

```bash
python main.py
```

This will:

1. Load the citizen registry from `data/citizens.json`.
2. Construct the city and its 10 streets.
3. Execute the full simulation loop for the configured number of days.
4. Persist results to MySQL (if configured).
5. Write a timestamped summary JSON to `data/logs/`.

### 6. Inspect a Citizen (Optional)

```bash
python spy_agent.py
```

Launch the interactive terminal inspector to browse any citizen's lifecycle, memory timeline, and moral choices.

### 7. Run a Quick Demo (Optional)

```bash
python test_run.py
```

Runs a short 10-day simulation seeded with a `drought_rumor` narrative event.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open an issue first to discuss any proposed changes, and ensure that your code follows the existing style conventions and includes appropriate tests.

---

<p align="center"><i>Built for emergent worlds. 🌍</i></p>
