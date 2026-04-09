"""Player and team name normalization utilities."""

import json
import unicodedata
from pathlib import Path
from functools import lru_cache


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@lru_cache(maxsize=1)
def _load_team_mapping():
    with open(CONFIG_DIR / "team_mapping.json", encoding="utf-8") as f:
        return json.load(f)


def normalize_team_name(name: str) -> str:
    """Normalize a team name to the canonical short form used in epl_final.csv."""
    mapping = _load_team_mapping()

    # Direct match on canonical name
    if name in mapping:
        return name

    # Search through aliases
    for canonical, aliases in mapping.items():
        if name in aliases:
            return canonical

    # Fallback: return as-is
    return name.strip()


def normalize_player_name(name: str) -> str:
    """Normalize a player name for consistent matching across sources.

    - Strips accents (é -> e)
    - Lowercases
    - Strips extra whitespace
    """
    # Decompose unicode, remove combining characters (accents)
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase and clean whitespace
    return " ".join(stripped.lower().split())
