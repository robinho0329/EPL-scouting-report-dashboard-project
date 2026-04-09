"""EPL Project Global Settings"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FBREF_RAW_DIR = RAW_DIR / "fbref"
PROCESSED_DIR = DATA_DIR / "processed"
DASHBOARD_DIR = DATA_DIR / "dashboard"
CONFIG_DIR = PROJECT_ROOT / "config"

# Existing match-level CSV
MATCH_CSV = PROJECT_ROOT / "epl_final.csv"

# SQLite DB for crawled player data
PLAYER_DB = PROCESSED_DIR / "epl_players.db"

# ============================================================
# Crawling Settings
# ============================================================
FBREF_BASE_URL = "https://fbref.com"
FBREF_MIN_INTERVAL = 6.0          # seconds between requests (FBref rate limit)
FBREF_MAX_RETRIES = 3
FBREF_BACKOFF_FACTOR = 2.0        # exponential backoff multiplier
FBREF_TIMEOUT = 30                # request timeout in seconds

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ============================================================
# Transfermarkt Settings
# ============================================================
TM_BASE_URL = "https://www.transfermarkt.com"
TM_RAW_DIR = RAW_DIR / "transfermarkt"
TM_MIN_INTERVAL = 5.0             # seconds between requests
TM_LEAGUE_ID = "GB1"              # Premier League
TM_LEAGUE_PATH = "premier-league"

def tm_season_id(season: str) -> int:
    """Convert '2000/01' -> 2000 (Transfermarkt uses start year)"""
    return int(season.split("/")[0])

# ============================================================
# Seasons
# ============================================================
SEASONS = [f"{y}/{str(y+1)[-2:]}" for y in range(2000, 2025)]
# e.g., ['2000/01', '2001/02', ..., '2024/25']

# FBref season URL IDs (season -> fbref path segment)
# FBref uses format: /en/comps/9/{season_id}/stats/
# The season_id is like "2000-2001", "2001-2002", etc.
def fbref_season_id(season: str) -> str:
    """Convert '2000/01' -> '2000-2001'"""
    start_year = int(season.split("/")[0])
    end_year = start_year + 1
    return f"{start_year}-{end_year}"

# ============================================================
# Checkpoint DB (for resume support)
# ============================================================
CHECKPOINT_DB = DATA_DIR / "crawl_checkpoint.db"

# ============================================================
# Logging
# ============================================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "INFO"
