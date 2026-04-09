"""FBref Crawler Agent - Selenium-based EPL player statistics crawler.

Crawling strategy:
1. Season squad stats page -> extract team links (1 request per season)
2. Per-team squad page -> player season stats + matchlog URLs (20 requests per season)
3. Player match logs -> individual match-level stats (N requests per team)

FBref URL patterns:
- Season stats: /en/comps/9/{season_id}/stats/{season_id}-Premier-League-Stats
- Team season: /en/squads/{team_id}/{season_id}/
- Player match log: /en/players/{player_id}/matchlogs/{season_id}/summary/
"""

import re
import logging
from pathlib import Path
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup, Comment

from config.settings import (
    FBREF_BASE_URL,
    FBREF_RAW_DIR,
    FBREF_MIN_INTERVAL,
    fbref_season_id,
)
from crawlers.base_agent import BaseCrawlerAgent
from crawlers.utils.name_normalizer import normalize_team_name

logger = logging.getLogger(__name__)

# 매치로그 크롤링할 최소 출전 경기수
MIN_APPEARANCES_FOR_MATCHLOG = 5


class FBrefAgent(BaseCrawlerAgent):
    """Crawls player statistics from FBref.com using Selenium."""

    def __init__(self):
        super().__init__(source_name="fbref", min_interval=FBREF_MIN_INTERVAL)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def crawl_season(self, season: str):
        """Crawl all player data for a given season (e.g., '2000/01').

        Steps:
        1. Get season overview page -> extract team links
        2. For each team -> get squad page -> extract player season stats
        3. For each player (with 5+ appearances) -> get match log
        """
        season_id = fbref_season_id(season)
        logger.info(f"=== Starting crawl for season {season} ({season_id}) ===")

        # Step 1: Get team links from season page
        teams = self._get_season_teams(season, season_id)
        if not teams:
            logger.error(f"Failed to get teams for {season}")
            return

        logger.info(f"Found {len(teams)} teams for {season}")

        # Step 2 & 3: Crawl each team's squad stats + player match logs
        for team_name, team_url in teams.items():
            task_key = f"team_{season_id}_{team_name}"
            if self.is_completed(task_key):
                logger.info(f"Skipping (already done): {team_name} {season}")
                continue

            try:
                self._crawl_team_season(season, season_id, team_name, team_url)
                self.mark_completed(task_key)
            except Exception as e:
                logger.error(f"Error crawling {team_name} {season}: {e}")
                self.mark_failed(task_key, str(e))

        logger.info(f"=== Completed season {season} ===")
        progress = self.get_progress()
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        logger.info(f"  Progress: {completed} completed, {failed} failed ({completed + failed} total)")

    def crawl_all_seasons(self, seasons=None):
        """Crawl all specified seasons sequentially."""
        from config.settings import SEASONS
        seasons = seasons or SEASONS

        try:
            for season in seasons:
                self.crawl_season(season)
        finally:
            self.close()

    # ==================================================================
    # SEASON LEVEL
    # ==================================================================

    def _get_season_teams(self, season, season_id):
        """Get all team names and their FBref URLs for a season.
        Returns: {team_name: team_url}
        """
        url = f"{FBREF_BASE_URL}/en/comps/9/{season_id}/stats/{season_id}-Premier-League-Stats"
        soup = self.fetch(url)
        if not soup:
            return {}

        teams = {}

        # Find the stats table - look for squad links
        for link in soup.select(
            "table.stats_table th[data-stat='team'] a, "
            "table.stats_table td[data-stat='team'] a"
        ):
            team_name = link.get_text(strip=True)
            team_url = FBREF_BASE_URL + link["href"]
            teams[team_name] = team_url

        # Fallback: search in any table for squad links
        if not teams:
            for link in soup.find_all("a", href=re.compile(r"/en/squads/")):
                href = link["href"]
                if season_id in href or "/stats/" in href:
                    team_name = link.get_text(strip=True)
                    if team_name and len(team_name) > 2:
                        team_url = FBREF_BASE_URL + href
                        teams[team_name] = team_url

        # "vs TeamName" 제거 - FBref 상대팀 행
        teams = {k: v for k, v in teams.items() if not k.startswith("vs ")}

        return teams

    # ==================================================================
    # TEAM LEVEL
    # ==================================================================

    def _crawl_team_season(
        self, season: str, season_id: str, team_name: str, team_url: str
    ):
        """Crawl a team's squad page and extract player stats + player match logs."""
        logger.info(f"Crawling team: {team_name} ({season})")

        soup = self.fetch(team_url)
        if not soup:
            return

        # Parse standard stats table
        table = self._find_stats_table(soup, "stats_standard")
        if table is None:
            logger.warning(f"No standard stats table found for {team_name} {season}")
            return

        players_data = self._parse_squad_stats(table, season, team_name)

        # Save team data
        save_dir = FBREF_RAW_DIR / season_id / normalize_team_name(team_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        if players_data:
            df = pd.DataFrame(players_data)
            df.to_csv(save_dir / "squad_stats.csv", index=False, encoding="utf-8-sig")
            logger.info(f"Saved {len(df)} players for {team_name} {season}")

            # Step 3: Extract matchlog URLs directly from the table
            matchlog_urls = self._extract_matchlog_links(table)
            logger.info(f"Found {len(matchlog_urls)} matchlog links for {team_name}")

            for player_name, matchlog_url in matchlog_urls.items():
                # Check if player has enough appearances
                player_row = df[df["player"] == player_name]
                if not player_row.empty:
                    mp_val = player_row.iloc[0].get("mp", 0)
                    try:
                        mp_val = int(float(mp_val))
                    except (ValueError, TypeError):
                        mp_val = 0

                    if mp_val >= MIN_APPEARANCES_FOR_MATCHLOG:
                        self._crawl_player_matchlog(
                            season, season_id, team_name,
                            player_name, matchlog_url
                        )
                    else:
                        logger.debug(f"Skipping {player_name} (mp={mp_val} < {MIN_APPEARANCES_FOR_MATCHLOG})")

    def _parse_squad_stats(
        self, table, season: str, team_name: str
    ):
        """Parse the standard stats table from a team's squad page."""
        players = []

        try:
            html_str = str(table)
            dfs = pd.read_html(StringIO(html_str))
            if not dfs:
                return players

            df = dfs[0]

            # Handle multi-level column headers - 하위 레벨만 사용
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[-1] for col in df.columns]

            # Clean column names
            df.columns = [self._clean_column_name(c) for c in df.columns]

            # 중복 컬럼명 처리 (같은 이름이 여러 개일 때)
            seen = {}
            new_cols = []
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols

            # Remove summary rows
            if "player" in df.columns:
                exclude = ["Squad Total", "Opponent Total"]
                df = df[df["player"].notna() & ~df["player"].isin(exclude)].copy()

            df["season"] = season
            df["team"] = team_name

            players = df.to_dict("records")

        except Exception as e:
            logger.error(f"Error parsing stats table for {team_name}: {e}")

        return players

    def _find_stats_table(self, soup: BeautifulSoup, table_id_part: str):
        """Find a stats table, checking both visible tables and HTML comments."""
        # Try direct search first
        table = soup.find("table", id=re.compile(table_id_part))
        if table:
            return table

        # FBref hides some tables in HTML comments for performance
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if table_id_part in comment:
                comment_soup = BeautifulSoup(comment, "lxml")
                table = comment_soup.find("table", id=re.compile(table_id_part))
                if table:
                    return table

        return None

    def _extract_matchlog_links(self, table):
        """Extract player name -> matchlog URL from the stats table.

        FBref provides matchlog links directly in td[data-stat='matches'].
        Player names are in th[data-stat='player'].
        """
        matchlog_urls = {}

        for row in table.find_all("tr"):
            th = row.find("th", attrs={"data-stat": "player"})
            td = row.find("td", attrs={"data-stat": "matches"})

            if th and td:
                a_player = th.find("a")
                a_matchlog = td.find("a")

                if a_player and a_matchlog:
                    name = a_player.get_text(strip=True)
                    href = a_matchlog.get("href", "")
                    if href and "/matchlogs/" in href:
                        matchlog_urls[name] = FBREF_BASE_URL + href

        return matchlog_urls

    # ==================================================================
    # PLAYER MATCH LOG LEVEL
    # ==================================================================

    def _crawl_player_matchlog(
        self,
        season: str,
        season_id: str,
        team_name: str,
        player_name: str,
        matchlog_url: str,
    ):
        """Crawl a player's match log for a specific season."""
        task_key = f"matchlog_{season_id}_{team_name}_{player_name}"
        if self.is_completed(task_key):
            return

        logger.info(f"  Crawling matchlog: {player_name} ({team_name} {season})")

        soup = self.fetch(matchlog_url)
        if not soup:
            self.mark_failed(task_key, "fetch failed")
            return

        matchlog_data = self._parse_matchlog(soup, season, team_name, player_name)

        if matchlog_data:
            save_dir = (
                FBREF_RAW_DIR / season_id / normalize_team_name(team_name) / "matchlogs"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            # Sanitize player name for filename
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", player_name)
            df = pd.DataFrame(matchlog_data)
            df.to_csv(save_dir / f"{safe_name}.csv", index=False, encoding="utf-8-sig")
            logger.info(f"  Saved matchlog: {player_name} - {len(df)} matches")
        else:
            logger.warning(f"  No matchlog data for {player_name}")

        self.mark_completed(task_key)

    def _parse_matchlog(
        self, soup: BeautifulSoup, season: str, team_name: str, player_name: str
    ):
        """Parse a player's match log page."""
        matches = []

        table = self._find_stats_table(soup, "matchlogs_all")
        if table is None:
            table = self._find_stats_table(soup, "matchlogs")
        if table is None:
            return matches

        try:
            html_str = str(table)
            dfs = pd.read_html(StringIO(html_str))
            if not dfs:
                return matches

            df = dfs[0]

            # Handle multi-level headers - 하위 레벨만 사용
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[-1] for col in df.columns]

            df.columns = [self._clean_column_name(c) for c in df.columns]

            # 중복 컬럼명 처리
            seen = {}
            new_cols = []
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols

            # Filter out non-Premier League matches and summary rows
            if "comp" in df.columns:
                df = df[df["comp"].str.contains("Premier League", case=False, na=False)]
            if "date" in df.columns:
                df = df[df["date"].notna()]

            df = df.copy()
            df["season"] = season
            df["team"] = team_name
            df["player"] = player_name

            matches = df.to_dict("records")

        except Exception as e:
            logger.error(f"Error parsing match log for {player_name}: {e}")

        return matches

    # ==================================================================
    # UTILITIES
    # ==================================================================

    @staticmethod
    def _clean_column_name(col: str) -> str:
        """Clean a column name to a consistent lowercase format."""
        col = str(col).lower().strip()
        for prefix in [
            "unnamed:", "level_0_", "playing time_", "performance_",
            "expected_", "progression_", "per 90 minutes_",
            "standard_", "shooting_", "passing_", "pass types_",
            "goal and shot creation_", "defensive actions_",
            "possession_", "miscellaneous stats_",
        ]:
            col = col.replace(prefix, "")
        col = re.sub(r"[^a-z0-9]+", "_", col).strip("_")
        return col


# ==================================================================
# CLI Entry Point
# ==================================================================

if __name__ == "__main__":
    import sys
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="FBref EPL Crawler")
    parser.add_argument("--season", type=str, help="Specific season (e.g., '2023/24')")
    parser.add_argument("--from-season", type=str, help="Start from this season")
    args = parser.parse_args()

    agent = FBrefAgent()
    try:
        if args.season:
            agent.crawl_season(args.season)
        elif args.from_season:
            from config.settings import SEASONS
            start_idx = SEASONS.index(args.from_season)
            agent.crawl_all_seasons(SEASONS[start_idx:])
        else:
            agent.crawl_all_seasons()
    finally:
        agent.close()
