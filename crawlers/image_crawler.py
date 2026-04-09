"""EPL 팀 로고 & 선수 이미지 크롤러

Transfermarkt에서 2000~2025시즌 EPL 팀 로고와 선수 이미지를 수집.
- HTML 페이지: undetected-chromedriver (Cloudflare 우회)
- 이미지 다운로드: requests (CDN은 봇 차단 없음)
- 레이트 리밋: 5초 (Transfermarkt 정책 준수)
- 체크포인트: SQLite로 중단 후 재개 지원

팀 로고 크롤링 방식:
  TM EPL 시즌 페이지(GB1)에서 연도별로 출전팀 목록과 TM verein ID를 직접 파싱.
  하드코딩 ID 방식 대신 TM 공식 데이터 사용 → 100% 정확한 로고 보장.
"""

import re
import sys
import time
import logging
import sqlite3
import hashlib
import requests
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (crawlers 패키지 임포트를 위해)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bs4 import BeautifulSoup
from crawlers.base_agent import BaseCrawlerAgent

# ── 경로 설정 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
IMG_DIR       = ROOT / "data" / "images"
TEAM_LOGO_DIR = IMG_DIR / "team_logos"
PLAYER_IMG_DIR = IMG_DIR / "players"
CHECKPOINT_DB  = ROOT / "data" / "image_crawl_checkpoint.db"
LOG_DIR        = ROOT / "logs"

IMG_DIR.mkdir(parents=True, exist_ok=True)
TEAM_LOGO_DIR.mkdir(parents=True, exist_ok=True)
PLAYER_IMG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── 상수 ───────────────────────────────────────────────
TM_BASE      = "https://www.transfermarkt.com"
TM_RATE      = 5.0   # 초 (Transfermarkt 레이트 리밋)
IMG_TIMEOUT  = 15

# EPL TM 시즌 페이지 (GB1 = Premier League)
EPL_SEASON_URL = (
    "https://www.transfermarkt.com/premier-league/startseite"
    "/wettbewerb/GB1/plus/?saison_id={year}"
)
EPL_SEASONS = range(2000, 2025)  # 2000/01 ~ 2024/25 시즌

# 이미지 다운로드용 헤더 (CDN은 Cloudflare 불필요)
IMG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.transfermarkt.com",
}

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "image_crawler.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("image_crawler")


# ── 체크포인트 DB ───────────────────────────────────────
def init_checkpoint_db():
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS downloaded (
            key TEXT PRIMARY KEY,
            local_path TEXT,
            url TEXT,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def is_downloaded(key: str) -> bool:
    conn = sqlite3.connect(CHECKPOINT_DB)
    row = conn.execute("SELECT local_path FROM downloaded WHERE key=?", (key,)).fetchone()
    conn.close()
    if row:
        return Path(row[0]).exists()
    return False


def mark_downloaded(key: str, local_path: str, url: str):
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.execute(
        "INSERT OR REPLACE INTO downloaded(key, local_path, url) VALUES(?,?,?)",
        (key, str(local_path), url)
    )
    conn.commit()
    conn.close()


# ── 이미지 다운로드 유틸 ────────────────────────────────
def download_image(url: str, save_path: Path) -> bool:
    """CDN에서 이미지 다운로드 (requests 사용, Cloudflare 불필요)."""
    try:
        resp = requests.get(url, headers=IMG_HEADERS, timeout=IMG_TIMEOUT)
        resp.raise_for_status()
        if len(resp.content) < 500:   # 빈 이미지 방어
            return False
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning(f"이미지 다운로드 실패: {url} — {e}")
        return False


# ── 팀 로고 크롤러 ──────────────────────────────────────
class TeamLogoCrawler(BaseCrawlerAgent):
    """TM EPL 시즌 페이지(GB1)를 순회해 출전팀 목록과 로고를 수집.

    하드코딩 ID 방식 대신, TM 공식 리그 페이지에서 verein ID와 로고 URL을
    직접 파싱하므로 국가 오류·ID 불일치 없이 100% EPL 팀만 수집.
    """

    def __init__(self):
        super().__init__(source_name="image_crawler", min_interval=TM_RATE)

    def _collect_teams_from_seasons(self, driver) -> dict:
        """EPL_SEASONS를 순회하며 {tm_id: (team_name, logo_url)} 수집.

        파싱 전략:
        - 팀명: a[href*='/startseite/verein/'] 에서 숫자·통화 제외한 텍스트
        - 로고: img[src*='wappen'] 에서 verein ID 추출 후 head 사이즈로 변환
        - 두 결과를 tm_id 기준으로 결합
        """
        seen = {}  # tm_id → (team_name, logo_url)

        for year in EPL_SEASONS:
            url = EPL_SEASON_URL.format(year=year)
            logger.info(f"[시즌] {year}/{str(year+1)[-2:]} 페이지 파싱 중...")
            try:
                self.rate_limiter.wait()
                driver.get(url)
                time.sleep(2)
                soup = BeautifulSoup(driver.page_source, "html.parser")
            except Exception as e:
                logger.error(f"[FAIL] {year} 시즌 페이지 로드 실패: {e}")
                continue

            # 팀명 수집: /startseite/verein/ 링크에서 팀명만 추출
            # (kader, spieler 등 다른 경로 링크는 숫자·통화 텍스트라 제외됨)
            team_map = {}
            for a in soup.select("a[href*='/startseite/verein/']"):
                text = a.get_text(strip=True)
                href = a.get("href", "")
                m = re.search(r"/verein/(\d+)", href)
                if not m or not text:
                    continue
                # 숫자만 있는 텍스트(선수수), 통화 기호 포함 텍스트(시장가치) 제외
                if text.isdigit() or "€" in text or "£" in text or "$" in text:
                    continue
                tm_id = int(m.group(1))
                team_map[tm_id] = text

            # 로고 수집: wappen URL에서 verein ID 추출 후 head 사이즈 변환
            logo_map = {}
            for img in soup.select("img[src*='wappen']"):
                src = img.get("src", "")
                m = re.search(r"/wappen/\w+/(\d+)\.png", src)
                if not m:
                    continue
                logo_id = int(m.group(1))
                if logo_id in logo_map:
                    continue
                if src.startswith("//"):
                    src = "https:" + src
                # tiny/mini/normal → head (고해상도)
                logo_url = re.sub(r"/wappen/\w+/", "/wappen/head/", src)
                # lm= 파라미터 제거 (캐시 무효화 방지)
                logo_url = re.sub(r"\?lm=\d+", "", logo_url)
                logo_map[logo_id] = logo_url

            # 팀명 + 로고 결합 (tm_id 공통 키)
            new_count = 0
            for tm_id, team_name in team_map.items():
                if tm_id in seen:
                    continue
                logo_url = logo_map.get(tm_id)
                if logo_url:
                    seen[tm_id] = (team_name, logo_url)
                    new_count += 1

            logger.info(f"  → {new_count}개 신규 팀 발견 (누적 {len(seen)}개)")

        return seen

    def crawl(self):
        driver = self._get_driver()

        # 1단계: 모든 시즌에서 EPL 팀 목록 수집
        logger.info(f"=== EPL 시즌 페이지 파싱 시작 ({min(EPL_SEASONS)}~{max(EPL_SEASONS)}) ===")
        teams = self._collect_teams_from_seasons(driver)
        logger.info(f"=== 총 {len(teams)}개 EPL 팀 발견 → 로고 다운로드 시작 ===")

        # 2단계: 로고 다운로드
        success, skip, fail = 0, 0, 0
        for tm_id, (team_name, logo_url) in sorted(teams.items(), key=lambda x: x[1][0]):
            key = f"logo_{tm_id}"
            safe_name = re.sub(r"[^\w\s-]", "", team_name).strip().replace(" ", "_")
            save_path = TEAM_LOGO_DIR / f"{safe_name}.png"

            if is_downloaded(key):
                logger.info(f"[SKIP] {team_name}")
                skip += 1
                continue

            if download_image(logo_url, save_path):
                mark_downloaded(key, save_path, logo_url)
                logger.info(f"[OK] {team_name} (ID={tm_id}) → {save_path.name}")
                success += 1
            else:
                logger.warning(f"[FAIL] {team_name} (ID={tm_id}) 다운로드 실패")
                fail += 1

        logger.info(f"=== 팀 로고 완료 — 성공: {success}, 스킵: {skip}, 실패: {fail} ===")
        self.close()


# ── 선수 이미지 크롤러 ──────────────────────────────────
class PlayerImageCrawler(BaseCrawlerAgent):
    """undetected-chromedriver로 Transfermarkt 검색 페이지에서 선수 이미지 수집."""

    def __init__(self):
        super().__init__(source_name="image_crawler", min_interval=TM_RATE)

    def crawl(self, player_df, max_players: int = None):
        import urllib.parse

        players = (
            player_df.drop_duplicates("player").head(max_players)
            if max_players else
            player_df.drop_duplicates("player")
        )
        logger.info(f"=== 선수 이미지 크롤링 시작 (총 {len(players)}명) ===")
        driver = self._get_driver()
        success, skip, fail = 0, 0, 0

        for _, row in players.iterrows():
            player_name = row["player"]
            key = f"player_{hashlib.md5(player_name.encode()).hexdigest()[:8]}_{player_name}"
            safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in player_name)
            save_path = PLAYER_IMG_DIR / f"{safe_name}.jpg"

            if is_downloaded(key):
                skip += 1
                continue

            # 검색 쿼리 후보 생성
            # 1순위: 원본 이름, 2순위: 이름 순서 뒤집기 (성→이름 순 처리, 예: "Son Heung-min" → "Heung-min Son")
            parts = player_name.split()
            if len(parts) >= 2:
                reversed_name = " ".join(parts[1:] + parts[:1])
            else:
                reversed_name = None
            search_candidates = [player_name]
            if reversed_name and reversed_name != player_name:
                search_candidates.append(reversed_name)

            img_url = None
            used_query = None

            for query_name in search_candidates:
                query = urllib.parse.quote(query_name)
                search_url = f"{TM_BASE}/schnellsuche/ergebnis/schnellsuche?query={query}&x=0&y=0"

                try:
                    self.rate_limiter.wait()
                    driver.get(search_url)
                    time.sleep(3)
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                except Exception as e:
                    logger.warning(f"[FAIL] {player_name} 검색 페이지 오류: {e}")
                    break

                # 디버그: 첫 번째 선수에서 페이지 소스 일부 저장
                if success + fail == 0:
                    debug_path = ROOT / "logs" / "tm_search_debug.html"
                    debug_path.write_text(driver.page_source[:50000], encoding="utf-8")
                    logger.info(f"[DEBUG] 첫 번째 검색 페이지 소스 저장: {debug_path}")
                    all_imgs = [img.get("src", "") or img.get("data-src", "") for img in soup.find_all("img")]
                    logger.info(f"[DEBUG] 페이지 내 img 태그 수: {len(all_imgs)}")
                    for s in all_imgs[:20]:
                        if s:
                            logger.info(f"[DEBUG]   img src: {s[:120]}")

                # 이미지 탐색: 검색 결과 테이블 첫 번째 행에서만 추출
                # (페이지 전체 portrait 이미지를 긁으면 엉뚱한 선수 이미지가 나옴)
                first_row = soup.select_one("table.items tbody tr")
                if first_row:
                    for img in first_row.select("img"):
                        src = img.get("src") or img.get("data-src", "")
                        if not src or src in ("", "about:blank") or "blank" in src:
                            continue
                        if "default" in src.lower():
                            continue
                        # portrait/header 또는 portrait/small URL만 허용 (TM 선수 이미지 패턴)
                        if "portrait" not in src:
                            continue
                        img_url = src
                        if img_url.startswith("//"):
                            img_url = "https:" + img_url
                        used_query = query_name
                        logger.info(f"[DEBUG] 첫 번째 행 이미지, 쿼리: '{query_name}', URL: {img_url[:100]}")
                        break

                if img_url:
                    break
                # 1순위 쿼리 실패 시 이름 뒤집어 재시도 로그
                if query_name == player_name and len(search_candidates) > 1:
                    logger.info(f"[RETRY] '{player_name}' 검색 실패 → '{reversed_name}' 재시도")

            if not img_url:
                logger.warning(f"[FAIL] {player_name} — 이미지 URL 없음 (모든 쿼리/셀렉터 미매칭)")
                fail += 1
                continue

            if download_image(img_url, save_path):
                mark_downloaded(key, save_path, img_url)
                retry_tag = f" (쿼리: '{used_query}')" if used_query != player_name else ""
                logger.info(f"[OK] {player_name}{retry_tag}")
                success += 1
            else:
                fail += 1

            if (success + fail) % 20 == 0:
                logger.info(f"  진행 중 — 성공: {success}, 스킵: {skip}, 실패: {fail}")

        logger.info(f"=== 선수 이미지 완료 — 성공: {success}, 스킵: {skip}, 실패: {fail} ===")
        self.close()


# ── 진행 현황 리포트 ────────────────────────────────────
def report_status():
    logos   = list(TEAM_LOGO_DIR.glob("*.png")) + list(TEAM_LOGO_DIR.glob("*.jpg"))
    players = list(PLAYER_IMG_DIR.glob("*.jpg")) + list(PLAYER_IMG_DIR.glob("*.png"))
    print("\n📊 이미지 크롤링 현황")
    print(f"  팀 로고:     {len(logos):>4}개  ({TEAM_LOGO_DIR})")
    print(f"  선수 이미지: {len(players):>4}개  ({PLAYER_IMG_DIR})")
    conn   = sqlite3.connect(CHECKPOINT_DB)
    total  = conn.execute("SELECT COUNT(*) FROM downloaded").fetchone()[0]
    conn.close()
    print(f"  DB 체크포인트: {total}건\n")


# ── 메인 실행 ───────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="EPL 이미지 크롤러 (undetected-chromedriver)")
    parser.add_argument("--mode", choices=["logos", "players", "both", "status"],
                        default="both", help="크롤링 모드")
    parser.add_argument("--max-players", type=int, default=None,
                        help="최대 선수 수 (테스트용, 예: --max-players 50)")
    parser.add_argument("--data-path", type=str,
                        default=str(ROOT / "data" / "processed" / "player_season_stats.parquet"),
                        help="선수 목록 parquet 경로 (player 컬럼 필수)")
    args = parser.parse_args()

    init_checkpoint_db()

    if args.mode == "status":
        report_status()

    elif args.mode == "logos":
        TeamLogoCrawler().crawl()
        report_status()

    elif args.mode == "players":
        try:
            df = pd.read_parquet(args.data_path)
            PlayerImageCrawler().crawl(df, max_players=args.max_players)
        except FileNotFoundError:
            logger.error(f"선수 데이터 파일 없음: {args.data_path}")
        report_status()

    elif args.mode == "both":
        TeamLogoCrawler().crawl()
        try:
            df = pd.read_parquet(args.data_path)
            PlayerImageCrawler().crawl(df, max_players=args.max_players)
        except FileNotFoundError:
            logger.error(f"선수 데이터 파일 없음: {args.data_path}")
        report_status()
