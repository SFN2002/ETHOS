#!/usr/bin/env python3
"""
spy_agent.py — Terminal inspector for a single ETHOS citizen lifecycle.

Usage:
    python spy_agent.py

The script discovers the most recent simulation log in data/logs/, prompts for
a citizen ID or profession, resolves ambiguous profession matches, and prints a
rich, color-coded profile plus a chronological memory-stream timeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output even on Windows terminals or when piped
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# -----------------------------------------------------------------------------
# Terminal styling
# -----------------------------------------------------------------------------
class Color:
    """ANSI escape codes for rich terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_MAGENTA = "\033[45m"


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
CITIZENS_FILE = DATA_DIR / "citizens.json"

WIDTH = 76


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Load and parse a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_citizens_registry() -> dict[int, dict[str, Any]]:
    """Load the raw citizen registry keyed by citizen id."""
    if not CITIZENS_FILE.exists():
        return {}
    records = load_json(CITIZENS_FILE)
    if not isinstance(records, list):
        return {}
    return {record["id"]: record for record in records if isinstance(record, dict)}


# -----------------------------------------------------------------------------
# Log discovery
# -----------------------------------------------------------------------------
def discover_latest_log() -> Path:
    """
    Return the most recently modified simulation log in data/logs/.

    Supports both current naming (simulation_YYYYMMDD_HHMMSS.json) and a
    possible future naming convention (simulation_day_XX.json).
    """
    if not LOGS_DIR.exists():
        raise FileNotFoundError(
            f"Logs directory not found: {LOGS_DIR}\n"
            "Run a simulation first to generate output logs."
        )

    candidates: list[Path] = []
    for pattern in ("simulation_*.json", "simulation_day_*.json"):
        candidates.extend(LOGS_DIR.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No simulation logs found in {LOGS_DIR}\n"
            "Run a simulation first to generate output logs."
        )

    # Most recently modified file wins
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------------------------------------------------------
# Search & ambiguity resolution
# -----------------------------------------------------------------------------
def find_matching_agents(
    log_data: dict[str, Any], query: str
) -> list[dict[str, Any]]:
    """Find agents by numeric id or profession substring (case-insensitive)."""
    query = query.strip()
    summaries = log_data.get("agent_summaries", [])

    if not query:
        return []

    if query.isdigit():
        target_id = int(query)
        return [agent for agent in summaries if agent.get("id") == target_id]

    query_lower = query.lower()
    return [
        agent
        for agent in summaries
        if query_lower in str(agent.get("profession", "")).lower()
    ]


def prompt_user(prompt: str) -> str:
    """Wrap input() so the script can be tested or redirected cleanly."""
    return input(prompt)


def resolve_ambiguity(
    candidates: list[dict[str, Any]], registry: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Display a numbered list and let the user pick the exact citizen."""
    print(
        f"\n{Color.BOLD}{Color.YELLOW}"
        f"[INFO] Multiple citizens matched your search. Please select one:{Color.RESET}"
    )
    print(f"{Color.DIM}{'─' * WIDTH}{Color.RESET}")

    for idx, agent in enumerate(candidates, 1):
        reg = registry.get(agent.get("id"), {})
        initial_wealth = reg.get("wealth", "N/A")
        print(
            f"{Color.BOLD}[{idx}]{Color.RESET} "
            f"{Color.CYAN}ID {agent.get('id'):>3}{Color.RESET} | "
            f"{Color.GREEN}{agent.get('name', 'Unknown'):<12}{Color.RESET} | "
            f"{Color.BLUE}{agent.get('profession', 'Unknown'):<20}{Color.RESET} | "
            f"{Color.YELLOW}Initial Wealth: {initial_wealth}{Color.RESET}"
        )

    print(f"{Color.DIM}{'─' * WIDTH}{Color.RESET}")

    while True:
        choice = prompt_user(
            f"{Color.BOLD}Enter number (1-{len(candidates)}): {Color.RESET}"
        ).strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        print(
            f"{Color.RED}Invalid selection. Please enter a number between "
            f"1 and {len(candidates)}.{Color.RESET}"
        )


# -----------------------------------------------------------------------------
# Visual output builders
# -----------------------------------------------------------------------------
def _rule(char: str = "═") -> str:
    """Return a horizontal rule."""
    return char * WIDTH


def _block_header(title: str, color: str = Color.BG_BLUE) -> None:
    """Print a colored block header."""
    padded = f" {title} ".center(WIDTH)
    print(f"{Color.BOLD}{Color.WHITE}{color}{padded}{Color.RESET}")


def _kv_line(label: str, value: str, label_color: str = Color.CYAN) -> None:
    """Print a labeled key-value line."""
    print(f"  {label_color}{Color.BOLD}{label:<16}{Color.RESET} {value}")


def print_profile(
    agent: dict[str, Any], registry: dict[int, dict[str, Any]]
) -> None:
    """Render the citizen's core profile."""
    reg = registry.get(agent.get("id"), {})

    status = reg.get("status", "unknown")
    sons = reg.get("sons", 0)
    daughters = reg.get("daughters", 0)
    initial_wealth = reg.get("wealth", agent.get("wealth", 0.0))

    print()
    _block_header("AGENT CORE PROFILE", Color.BG_BLUE)
    print(_rule("─"))

    _kv_line("Name", f"{Color.GREEN}{agent.get('name', 'Unknown')}{Color.RESET}")
    _kv_line("Citizen ID", f"{Color.YELLOW}{agent.get('id', '?')}{Color.RESET}")
    _kv_line(
        "Profession", f"{Color.BLUE}{agent.get('profession', 'Unknown')}{Color.RESET}"
    )
    _kv_line("Family Size", f"{agent.get('family_size', '?')} members")
    _kv_line(
        "Status",
        f"{Color.MAGENTA}{status.capitalize()}{Color.RESET} "
        f"({Color.CYAN}{sons}{Color.RESET} son{'s' if sons != 1 else ''}, "
        f"{Color.CYAN}{daughters}{Color.RESET} daughter{'s' if daughters != 1 else ''})",
    )
    _kv_line(
        "Initial Wealth",
        f"{Color.YELLOW}{float(initial_wealth):.2f}{Color.RESET} coins",
    )
    _kv_line(
        "Final Wealth",
        f"{Color.YELLOW}{float(agent.get('wealth', 0.0)):.2f}{Color.RESET} coins",
    )
    _kv_line(
        "Final Happiness",
        f"{Color.GREEN}{float(agent.get('happiness', 0.0)):.2f}{Color.RESET}",
    )
    _kv_line(
        "Final Integrity",
        f"{Color.GREEN}{float(agent.get('integrity', 0.0)):.2f}{Color.RESET}",
    )
    _kv_line(
        "Religion",
        f"{Color.MAGENTA}{agent.get('religion', 'Undecided')}{Color.RESET}",
    )
    religion_reason = agent.get("religion_reason", "")
    if religion_reason:
        _kv_line(
            "Religion Reason",
            f"{Color.DIM}{religion_reason}{Color.RESET}",
        )
    _kv_line(
        "Dominant Emotion",
        f"{Color.CYAN}{agent.get('current_emotion', 'neutral')}{Color.RESET}",
    )
    _kv_line(
        "Psych. Tension",
        f"{Color.YELLOW}{float(agent.get('psychological_tension', 0.0)):.2f}{Color.RESET}",
    )
    _kv_line(
        "Last Action",
        f"{Color.GREEN}{agent.get('last_action_type', 'N/A')}{Color.RESET}",
    )
    action_details = agent.get("last_action_details", "")
    if action_details:
        _kv_line(
            "Action Details",
            f"{Color.DIM}{action_details}{Color.RESET}",
        )
    reconstructed_logic = agent.get("last_reconstructed_logic", "")
    if reconstructed_logic:
        _kv_line(
            "Reconstructed Logic",
            f"{Color.DIM}{reconstructed_logic}{Color.RESET}",
        )

    if agent.get("has_accepted_wage_sacrifice_deal", False):
        _kv_line(
            "DYSTOPIAN DEAL",
            f"{Color.RED}ACCEPTED{Color.RESET}",
        )
        _kv_line(
            "Sacrificed",
            f"{Color.RED}{agent.get('sacrificed_family_member', 'unknown')}{Color.RESET}",
        )
        _kv_line(
            "Wage Multiplier",
            f"{Color.YELLOW}{float(agent.get('dystopian_wage_multiplier', 1.0)):.1f}x{Color.RESET}",
        )
        decision = agent.get("dystopian_decision") or {}
        justification = decision.get("utilitarian_justification", "")
        if justification:
            _kv_line(
                "Dark Justification",
                f"{Color.DIM}{justification}{Color.RESET}",
            )

    print(_rule("─"))


def print_timeline(agent: dict[str, Any]) -> None:
    """Render the citizen's chronological memory-stream timeline."""
    memories = agent.get("memory_stream", [])

    print()
    _block_header(
        f"GENERATIVE MEMORY TIMELINE  •  {len(memories)} DAY(S)", Color.BG_MAGENTA
    )
    print()

    if not memories:
        print(
            f"  {Color.YELLOW}[WARN] No memory_stream entries found for this citizen.{Color.RESET}"
        )
        return

    for entry in sorted(memories, key=lambda m: m.get("day", 0)):
        day = entry.get("day", "?")
        event = entry.get(
            "event_description", "No event recorded."
        )
        family = entry.get(
            "family_status_snapshot", "No family snapshot recorded."
        )
        reflection = entry.get(
            "agent_reflection", "No reflection recorded."
        )
        emotion = entry.get("dominant_emotion", "")
        belief = entry.get("current_belief_system", "")
        action = entry.get("action_type", "")
        dystopian = entry.get("dystopian_decision")

        # Day header
        day_label = f" DAY {day} "
        print(
            f"{Color.BOLD}{Color.CYAN}+{day_label.center(WIDTH - 2, '-')}+{Color.RESET}"
        )

        # Wrapped body lines would be ideal, but we keep it simple and readable
        print(
            f"{Color.CYAN}|{Color.RESET} "
            f"{Color.BOLD}{Color.GREEN}Event:{Color.RESET}\n"
            f"{Color.CYAN}|{Color.RESET}   {event}"
        )
        if emotion or belief or action or dystopian:
            cognitive_bits = []
            if emotion:
                cognitive_bits.append(f"Emotion: {emotion}")
            if belief:
                cognitive_bits.append(f"Worldview: {belief}")
            if action:
                cognitive_bits.append(f"Action: {action}")
            if dystopian and dystopian.get("accept_deal"):
                cognitive_bits.append(
                    f"DYSTOPIAN DEAL: sacrificed {dystopian.get('abandoned_family_member', 'unknown')}"
                )
            print(
                f"{Color.CYAN}|{Color.RESET}\n"
                f"{Color.CYAN}|{Color.RESET} "
                f"{Color.BOLD}{Color.RED}Cognitive State:{Color.RESET}\n"
                f"{Color.CYAN}|{Color.RESET}   {', '.join(cognitive_bits)}"
            )
        print(
            f"{Color.CYAN}|{Color.RESET}\n"
            f"{Color.CYAN}|{Color.RESET} "
            f"{Color.BOLD}{Color.YELLOW}Family Status:{Color.RESET}\n"
            f"{Color.CYAN}|{Color.RESET}   {family}"
        )
        print(
            f"{Color.CYAN}|{Color.RESET}\n"
            f"{Color.CYAN}|{Color.RESET} "
            f"{Color.BOLD}{Color.MAGENTA}Reflection:{Color.RESET}\n"
            f"{Color.CYAN}|{Color.RESET}   {reflection}"
        )
        print(
            f"{Color.BOLD}{Color.CYAN}+{'-' * (WIDTH - 2)}+{Color.RESET}\n"
        )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run the spy_agent CLI."""
    try:
        log_path = discover_latest_log()
        print(
            f"\n{Color.GREEN}[OK] Latest simulation log:{Color.RESET} "
            f"{Color.DIM}{log_path}{Color.RESET}"
        )

        log_data = load_json(log_path)
        if not isinstance(log_data, dict) or "agent_summaries" not in log_data:
            print(
                f"{Color.RED}[ERR] Invalid log format: missing 'agent_summaries'.{Color.RESET}",
                file=sys.stderr,
            )
            return 1

        registry = load_citizens_registry()

        query = prompt_user(
            f"\n{Color.BOLD}Enter Citizen ID or Profession: {Color.RESET}"
        ).strip()

        if not query:
            print(f"{Color.RED}[ERR] No search term provided.{Color.RESET}")
            return 1

        matches = find_matching_agents(log_data, query)

        if not matches:
            print(
                f"{Color.RED}[ERR] No citizen found matching '{query}'.{Color.RESET}"
            )
            return 1

        if len(matches) == 1:
            agent = matches[0]
            print(
                f"\n{Color.GREEN}[OK] Found exactly one match:{Color.RESET} "
                f"{Color.BOLD}{agent.get('name')} ({agent.get('profession')}){Color.RESET}"
            )
        else:
            agent = resolve_ambiguity(matches, registry)

        print_profile(agent, registry)
        print_timeline(agent)

        return 0

    except FileNotFoundError as error:
        print(f"{Color.RED}[ERR] {error}{Color.RESET}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(
            f"{Color.RED}[ERR] Failed to parse JSON log: {error}{Color.RESET}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}[WARN] Operation cancelled by user.{Color.RESET}")
        return 130
    except Exception as error:
        print(
            f"{Color.RED}[ERR] Unexpected error: {error}{Color.RESET}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
