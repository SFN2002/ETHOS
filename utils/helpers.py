"""
Generic helper utilities for JSON I/O, descriptive statistics, and shared
simulation behaviour interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.agent import Agent


def load_json(path: str | Path) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        The parsed JSON payload.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}: {error}") from error


def save_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """
    Serialise ``data`` to JSON and write it to ``path``.

    Args:
        path: Destination filesystem path.
        data: Any JSON-serialisable object.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False, default=str)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp ``value`` to the inclusive interval [``lower``, ``upper``]."""
    return max(lower, min(upper, float(value)))


def describe(data: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    Compute descriptive statistics for a flat mapping of numeric values.

    Returns a dictionary keyed by the original metric name, each containing
    count, sum, mean, min, and max.
    """
    if not data:
        return {}

    summary: dict[str, dict[str, float]] = {}
    for key, value in data.items():
        numeric = float(value)
        summary[key] = {
            "count": 1.0,
            "sum": numeric,
            "mean": numeric,
            "min": numeric,
            "max": numeric,
        }
    return summary


def batch(items: list[Any], size: int) -> list[list[Any]]:
    """Split ``items`` into chunks of at most ``size``."""
    if size <= 0:
        raise ValueError("Batch size must be positive.")
    return [items[i : i + size] for i in range(0, len(items), size)]


def interpret_action_type(action_type: str | None) -> dict[str, Any]:
    """
    Convert a free-form action_type string into simulation consequences.

    No action is rejected: unmapped emergent actions are recorded as-is and
    receive a neutral economic signature.  Rebellious or non-standard labour
    actions modify income, fear, happiness, and integrity deltas.
    """
    if not action_type:
        return {
            "income_multiplier": 1.0,
            "fear_delta": 0.0,
            "happiness_delta": 0.0,
            "integrity_delta": 0.0,
            "rebellious": False,
            "nihilistic": False,
        }

    normalized = str(action_type).lower()

    # Rebellious / labour-withholding actions
    if any(word in normalized for word in ("strike", "protest", "riot", "refuse work")):
        return {
            "income_multiplier": 0.0,
            "fear_delta": 0.06,
            "happiness_delta": -0.03,
            "integrity_delta": 0.01,
            "rebellious": True,
            "nihilistic": False,
        }

    # Worldview collapse actions
    if any(
        word in normalized
        for word in ("nihilist", "nihilism", "atheist", "atheism", "agnostic")
    ):
        return {
            "income_multiplier": 0.9,
            "fear_delta": 0.02,
            "happiness_delta": -0.04,
            "integrity_delta": -0.02,
            "rebellious": False,
            "nihilistic": True,
        }

    # Anti-social / predatory actions
    if any(
        word in normalized
        for word in ("crime", "theft", "steal", "gamble", "fraud", "extort")
    ):
        return {
            "income_multiplier": 1.3,
            "fear_delta": 0.04,
            "happiness_delta": 0.02,
            "integrity_delta": -0.10,
            "rebellious": True,
            "nihilistic": False,
        }

    # Generous / altruistic actions
    if any(
        word in normalized
        for word in ("altruism", "donate", "charity", "give", "share", "help")
    ):
        return {
            "income_multiplier": 0.85,
            "fear_delta": -0.02,
            "happiness_delta": 0.04,
            "integrity_delta": 0.03,
            "rebellious": False,
            "nihilistic": False,
        }

    # Flight / migration actions
    if any(word in normalized for word in ("migrate", "flee", "leave", "escape")):
        return {
            "income_multiplier": 0.5,
            "fear_delta": 0.03,
            "happiness_delta": -0.02,
            "integrity_delta": 0.0,
            "rebellious": False,
            "nihilistic": False,
        }

    # Religious / ideological mobilization
    if any(
        word in normalized
        for word in ("cult", "preach", "convert", "crusade", "sermon")
    ):
        return {
            "income_multiplier": 0.75,
            "fear_delta": 0.03,
            "happiness_delta": 0.02,
            "integrity_delta": -0.01,
            "rebellious": False,
            "nihilistic": False,
        }

    # Default: preserve the emergent action without economic penalty.
    return {
        "income_multiplier": 1.0,
        "fear_delta": 0.0,
        "happiness_delta": 0.0,
        "integrity_delta": 0.0,
        "rebellious": False,
        "nihilistic": False,
    }


def build_spiritual_core(agent: "Agent") -> str:
    """Render the current (fluid) spiritual framework for prompt injection."""
    religion = getattr(agent, "religion", "Undecided")
    reason = getattr(agent, "religion_reason", "")
    if religion == "Undecided" or not reason:
        return (
            "[SPIRITUAL CORE: You have not yet committed to a worldview, or you "
            "have abandoned your previous one. You are free to adopt, reject, or "
            "invent any belief system today if your circumstances demand it.]\n\n"
        )
    return (
        f"[SPIRITUAL CORE: Your current worldview is {religion}. Your reason: "
        f"{reason} This framework currently guides you, but you are free to "
        "question, modify, or abandon it if it no longer justifies your suffering "
        "or aspirations.]\n\n"
    )
