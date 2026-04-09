"""Transfermarkt Crawler Agent - Market values, transfers, contract data.

Crawling strategy:
1. Season page -> get all team links + team market values
2. Team squad page (detailed) -> player market values, age, contract, nationality
3. Transfer page -> season transfers (arrivals + departures per team)

Transfermarkt URL patterns:
- Season teams: /premier-league/startseite/wettbewerb/GB1/plus/?saison_id={year}
- Team squad:   /{team-slug}/kader/verein/{team_id}/saison_id/{year}/plus/1
- Transfers:    /premier-league/transfers/wettbewerb/GB1/saison_id/{year}
- Player value: /{player-slug}/marktwertverlauf/spieler/{player_id}
"""

import re
import json
import logging
from pathlib import Path
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from config.settings import (
    TM_BASE_URL,
    TM_RAW_DIR,
    TM_MIN_INTERVAL,
    TM_LEAGUE_ID,
    TM_LEAGUE_PATH,
    tm_season_id,
)
from crawlers.base_agent import BaseCrawlerAgent

logger = logging.getLogger(__name__)


def parse_market_value(value_str: str) -> float:
    """Parse Transfermarkt market value string to float (in euros).

    Examples:
        '€85.00m' -> 85000000
        '€800k'   -> 800000
        '€1.20m'  -> 1200000
        '-'        -> 0
    """
    if not value_str or value_str == "-" or value_str == "?":
        return 0.0

    value_str = value_str.strip().replace("€", "").replace(",", ".")

    try:
        if "m" in value_str.lower():
            return float(value_str.lower().replace("m", "")) * 1_000_000
        elif "k" in value_str.lower():
            return float(value_str.lower().replace("k", "")) * 1_000
        elif "bn" in value_str.lower():
            return float(value_str.lower().replace("bn", "")) * 1_000_000_000
        else:
            return float(value_str)
    except (ValueError, TypeError):
        return 0.0


def parse_transfer_fee(fee_str: str) -> dict:
    """Parse transfer fee string.

    Returns dict with 'fee' (float) and 'type' (str: 'transfer', 'loan', 'free', 'unknown')
    """
    if not fee_str:
        return {"fee": 0.0, "type": "unknown"}

    fee_str = fee_str.strip()

    if "loan" in fee_str.lower():
        fee = parse_market_value(fee_str.replace("loan", "").replace("Loan", ""))
        return {"fee": fee, "type": "loan"}
    elif "free" in fee_str.lower() or fee_str == "-":
        return {"fee": 0.0, "type": "free"}
    elif "?" in fee_str or fee_str == "?":
        return {"fee": 0.0, "type": "unknown"}
    elif "End of loan" in fee_str:
        return {"fee": 0.0, "type": "end_of_loan"}
    else:
        return {"fee": parse_market_value(fee_str), "type": "transfer"}


class TransfermarktAgent(BaseCrawlerAgent):
    """Crawls market values and transfer data from Transfermarkt."""

    def __init__(self):
        super().__init__(source_name="transfermarkt", min_interval=TM_MIN_INTERVAL)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def crawl_season(self, season: str):
        """Crawl all Transfermarkt data for a given season.

        Steps:
        1. Get season overview -> team names, slugs, IDs, total market values
        2. For each team -> detailed squad page -> player market values
        3. Season transfers page -> all arrivals/departures
        """
        year = tm_season_id(season)
        logger.info(f"=== TM: Starting crawl for season {season} (saison_id={year}) ===")

        # Step 1: Get teams
        teams = self._get_season_teams(season, year)
        if not teams:
            logger.error(f"Failed to get teams for {season}")
            return

        logger.info(f"Found {len(teams)} teams for {season}")

        # Step 2: Crawl each team's squad values
        for team in teams:
            team_name = team["name"]
            task_key = f"tm_squad_{year}_{team_name}"

            if self.is_completed(task_key):
                logger.info(f"Skipping (done): {team_name} {season}")
                continue

            try:
                self._crawl_team_squad(season, year, team)
                self.mark_completed(task_key)
            except Exception as e:
                logger.error(f"Error crawling {team_name} {season}: {e}")
                self.mark_failed(task_key, str(e))

        # Step 3: Crawl transfers
        task_key = f"tm_transfers_{year}"
        if not self.is_completed(task_key):
            try:
                self._crawl_season_transfers(season, year)
                self.mark_completed(task_key)
            except Exception as e:
                logger.error(f"Error crawling transfers {season}: {e}")
                self.mark_failed(task_key, str(e))

        logger.info(f"=== TM: Completed season {season} ===")

    def crawl_all_seasons(self, seasons=None):
        """Crawl all specified seasons."""
        from config.settings import SEASONS
        seasons = seasons or SEASONS

        try:
            for season in seasons:
                self.crawl_season(season)
        finally:
            self.close()

    # ==================================================================
    # SEASON LEVEL - TEAMS
    # ==================================================================

    def _get_season_teams(self, season: str, year: int):
        """Get all teams with their IDs and slugs for a season.

        Returns: list of dicts with keys: name, slug, team_id, total_value
        """
        url = (
            f"{TM_BASE_URL}/{TM_LEAGUE_PATH}/startseite"
            f"/wettbewerb/{TM_LEAGUE_ID}/plus/?saison_id={year}"
        )
        soup = self.fetch(url)
        if not soup:
            return []

        teams = []
        table = soup.find("table", class_="items")
        if not table:
            logger.error(f"No teams table found for {season}")
            return []

        for row in table.find_all("tr", class_=["odd", "even"]):
            team_info = self._parse_team_row(row)
            if team_info:
                teams.append(team_info)

        # Save teams metadata
        save_dir = TM_RAW_DIR / str(year)
        save_dir.mkdir(parents=True, exist_ok=True)

        if teams:
            df = pd.DataFrame(teams)
            df["season"] = season
            df.to_csv(save_dir / "teams.csv", index=False, encoding="utf-8-sig")
            logger.info(f"Saved {len(teams)} teams metadata for {season}")

        return teams

    def _parse_team_row(self, row):
        """Parse a team row from the season overview table."""
        try:
            # Team name and link
            name_link = row.find("td", class_="hauptlink")
            if not name_link:
                return None

            a = name_link.find("a")
            if not a:
                return None

            name = a.get_text(strip=True)
            href = a.get("href", "")

            # Extract team_id and slug from URL
            # e.g., /fc-arsenal/startseite/verein/11/saison_id/2023
            match = re.search(r"/([^/]+)/startseite/verein/(\d+)", href)
            if not match:
                return None

            slug = match.group(1)
            team_id = match.group(2)

            # Squad size (찾을 수 있으면)
            squad_size = ""
            tds = row.find_all("td", class_="zentriert")
            for td in tds:
                text = td.get_text(strip=True)
                if text.isdigit() and 15 < int(text) < 60:
                    squad_size = text
                    break

            # Total market value
            value_td = row.find("td", class_="rechts")
            total_value_str = value_td.get_text(strip=True) if value_td else ""
            total_value = parse_market_value(total_value_str)

            return {
                "name": name,
                "slug": slug,
                "team_id": team_id,
                "squad_size": squad_size,
                "total_value": total_value,
                "total_value_str": total_value_str,
            }
        except Exception as e:
            logger.debug(f"Error parsing team row: {e}")
            return None

    # ==================================================================
    # TEAM SQUAD - PLAYER MARKET VALUES
    # ==================================================================

    def _crawl_team_squad(self, season: str, year: int, team: dict):
        """Crawl detailed squad page for player market values."""
        team_name = team["name"]
        slug = team["slug"]
        team_id = team["team_id"]

        # Detailed squad page (plus/1 for detailed view)
        url = f"{TM_BASE_URL}/{slug}/kader/verein/{team_id}/saison_id/{year}/plus/1"
        logger.info(f"TM: Crawling squad: {team_name} ({season})")

        soup = self.fetch(url)
        if not soup:
            return

        players = self._parse_squad_page(soup, season, team_name)

        if players:
            save_dir = TM_RAW_DIR / str(year) / self._safe_dirname(team_name)
            save_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(players)
            df.to_csv(save_dir / "squad_values.csv", index=False, encoding="utf-8-sig")
            logger.info(f"TM: Saved {len(df)} players for {team_name} {season}")

    def _parse_squad_page(self, soup, season, team_name):
        """Parse the detailed squad page for player data."""
        players = []

        table = soup.find("table", class_="items")
        if not table:
            return players

        for row in table.find_all("tr", class_=["odd", "even"]):
            player = self._parse_player_row(row, season, team_name)
            if player:
                players.append(player)

        return players

    def _parse_player_row(self, row, season, team_name):
        """Parse a single player row from squad page."""
        try:
            # 선수명 + 링크
            name_cell = row.find("td", class_="hauptlink")
            if not name_cell:
                return None

            a = name_cell.find("a")
            if not a:
                return None

            player_name = a.get_text(strip=True)
            player_href = a.get("href", "")

            # player_id 추출
            player_id = ""
            id_match = re.search(r"/spieler/(\d+)", player_href)
            if id_match:
                player_id = id_match.group(1)

            # 포지션 (posrela 클래스 아래 작은 텍스트)
            position = ""
            pos_td = row.find("td", class_="posrela")
            if pos_td:
                pos_small = pos_td.find_all("tr")
                if len(pos_small) > 1:
                    position = pos_small[-1].get_text(strip=True)
                else:
                    # 대안: inline-table 내부
                    pos_texts = pos_td.find_all("td")
                    if len(pos_texts) > 1:
                        position = pos_texts[-1].get_text(strip=True)

            # 모든 zentriert 셀에서 정보 추출
            centered_cells = row.find_all("td", class_="zentriert")
            dob = ""
            age = ""
            nationality = ""
            height = ""
            foot = ""
            joined = ""

            for cell in centered_cells:
                text = cell.get_text(strip=True)

                # 생년월일 + 나이 패턴: "DD/MM/YYYY (나이)" 또는 "15/09/1995 (29)"
                dob_age_match = re.match(
                    r"(\d{2}/\d{2}/\d{4})\s*\((\d{1,2})\)", text
                )
                if dob_age_match:
                    dob = dob_age_match.group(1)
                    age = dob_age_match.group(2)
                    continue

                # 대안 DOB 패턴: "Mar 7, 2001 (24)" 또는 "07.03.2001 (24)"
                if not dob:
                    alt_match = re.match(
                        r"(\w{3}\s+\d{1,2},\s+\d{4})\s*\((\d{1,2})\)", text
                    )
                    if alt_match:
                        dob = alt_match.group(1)
                        age = alt_match.group(2)
                        continue

                    alt_match2 = re.match(
                        r"(\d{2}\.\d{2}\.\d{4})\s*\((\d{1,2})\)", text
                    )
                    if alt_match2:
                        dob = alt_match2.group(1)
                        age = alt_match2.group(2)
                        continue

                # 키 (예: "1,83m")
                if "m" in text and "," in text and len(text) < 8:
                    height = text
                    continue

                # 발 (left/right/both)
                if text.lower() in ["left", "right", "both"]:
                    foot = text
                    continue

                # Joined 날짜 (예: "04/07/2024")
                if re.match(r"\d{2}/\d{2}/\d{4}$", text) and not dob:
                    joined = text
                elif re.match(r"\d{2}/\d{2}/\d{4}$", text) and dob and text != dob:
                    joined = text

                # 국적 (img의 title)
                imgs = cell.find_all("img", class_="flaggenrahmen")
                if imgs:
                    nationality = ", ".join(
                        img.get("title", "") for img in imgs if img.get("title")
                    )

            # 시가 (마지막 rechts hauptlink)
            value_td = row.find("td", class_="rechts hauptlink")
            market_value_str = value_td.get_text(strip=True) if value_td else ""
            market_value = parse_market_value(market_value_str)

            return {
                "player": player_name,
                "player_id": player_id,
                "position": position,
                "dob": dob,
                "age": age,
                "nationality": nationality,
                "height": height,
                "foot": foot,
                "joined": joined,
                "market_value": market_value,
                "market_value_str": market_value_str,
                "team": team_name,
                "season": season,
            }

        except Exception as e:
            logger.debug(f"Error parsing player row: {e}")
            return None

    # ==================================================================
    # TRANSFERS
    # ==================================================================

    def _crawl_season_transfers(self, season: str, year: int):
        """Crawl all transfers for a season."""
        url = (
            f"{TM_BASE_URL}/{TM_LEAGUE_PATH}/transfers"
            f"/wettbewerb/{TM_LEAGUE_ID}/saison_id/{year}"
        )
        logger.info(f"TM: Crawling transfers for {season}")

        soup = self.fetch(url)
        if not soup:
            return

        transfers = self._parse_transfers_page(soup, season)

        if transfers:
            save_dir = TM_RAW_DIR / str(year)
            save_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(transfers)
            df.to_csv(save_dir / "transfers.csv", index=False, encoding="utf-8-sig")
            logger.info(f"TM: Saved {len(df)} transfers for {season}")

    def _parse_transfers_page(self, soup, season):
        """Parse the transfers page for all arrivals and departures."""
        transfers = []

        # Transfermarkt groups transfers by team in boxes
        boxes = soup.find_all("div", class_="box")

        for box in boxes:
            # Team name from box header
            header = box.find("a", class_="vereinprofil_tooltip")
            if not header:
                continue
            team_name = header.get_text(strip=True)

            # Find arrivals and departures tables
            tables = box.find_all("table", class_="items")

            for i, table in enumerate(tables):
                direction = "in" if i == 0 else "out"

                for row in table.find_all("tr", class_=["odd", "even"]):
                    transfer = self._parse_transfer_row(
                        row, season, team_name, direction
                    )
                    if transfer:
                        transfers.append(transfer)

        return transfers

    def _parse_transfer_row(self, row, season, team_name, direction):
        """Parse a single transfer row."""
        try:
            # Player name
            name_cell = row.find("td", class_="hauptlink")
            if not name_cell:
                return None

            a = name_cell.find("a")
            if not a:
                return None

            player_name = a.get_text(strip=True)

            # Player ID
            player_id = ""
            href = a.get("href", "")
            id_match = re.search(r"/spieler/(\d+)", href)
            if id_match:
                player_id = id_match.group(1)

            # Age and position
            age = ""
            position = ""
            centered = row.find_all("td", class_="zentriert")
            for cell in centered:
                text = cell.get_text(strip=True)
                if text.isdigit() and 15 < int(text) < 45:
                    age = text
                # Position patterns
                if text in [
                    "Goalkeeper", "Centre-Back", "Left-Back", "Right-Back",
                    "Defensive Midfield", "Central Midfield", "Attacking Midfield",
                    "Left Winger", "Right Winger", "Centre-Forward",
                    "Left Midfield", "Right Midfield", "Second Striker",
                ]:
                    position = text

            # From/To club
            other_club = ""
            club_links = row.find_all("td", class_="vereinswappen_tooltip")
            if not club_links:
                club_links = row.find_all("a", class_="vereinprofil_tooltip")
            for cl in club_links:
                text = cl.get("title", "") or cl.get_text(strip=True)
                if text and text != team_name:
                    other_club = text
                    break

            # Transfer fee
            fee_td = row.find("td", class_="rechts")
            fee_str = fee_td.get_text(strip=True) if fee_td else ""
            fee_info = parse_transfer_fee(fee_str)

            # Market value at time of transfer
            mv_td = row.find("td", class_="rechts hauptlink")
            mv_str = mv_td.get_text(strip=True) if mv_td else ""

            return {
                "player": player_name,
                "player_id": player_id,
                "age": age,
                "position": position,
                "team": team_name,
                "direction": direction,  # "in" or "out"
                "other_club": other_club,
                "fee": fee_info["fee"],
                "fee_type": fee_info["type"],
                "fee_str": fee_str,
                "market_value_str": mv_str,
                "market_value": parse_market_value(mv_str),
                "season": season,
            }

        except Exception as e:
            logger.debug(f"Error parsing transfer row: {e}")
            return None

    # ==================================================================
    # UTILITIES
    # ==================================================================

    @staticmethod
    def _safe_dirname(name: str) -> str:
        """Make a safe directory name from team name."""
        return re.sub(r'[<>:"/\\|?*]', "_", name).strip().lower().replace(" ", "_")


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

    parser = argparse.ArgumentParser(description="Transfermarkt EPL Crawler")
    parser.add_argument("--season", type=str, help="Specific season (e.g., '2023/24')")
    parser.add_argument("--from-season", type=str, help="Start from this season")
    args = parser.parse_args()

    agent = TransfermarktAgent()
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
