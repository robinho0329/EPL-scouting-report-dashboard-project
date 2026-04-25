"""scripts/p8_crossleague_crawler.py — 해외 리그 → EPL 크로스리그 이적 데이터 수집.

P8 이적 적응 모델에 필요한 진짜 이적 케이스 수집:
  - 클럽 변경 + 리그 변경이 동시에 발생한 케이스
  - 대상: Bundesliga / La Liga / Serie A / Ligue 1 → EPL
  - 기간: 2009-10 ~ 2023-24 (15시즌)

설계 근거:
  - FBref 해외 리그 시즌 통계 페이지 1장 = 전 선수 포함 (확인됨)
  - data-stat 속성: player, team, position, age, games, games_starts,
                    minutes, minutes_90s, goals, assists (Bundesliga 2018-19 직접 확인)
  - EPL 이후 시즌 스탯은 기존 player_season_stats.parquet 활용
  - 매칭: 이름 정규화 후 외국 리그 시즌 → EPL 다음 시즌 등장 여부 확인

실행:
  python scripts/p8_crossleague_crawler.py

소요: 약 10~15분 (60페이지 × 6초 레이트리밋)
결과: data/raw/p8_crossleague_transfers.parquet
"""

import sys
import logging
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crawlers.base_agent import BaseCrawlerAgent

# ─── 경로 ────────────────────────────────────────────────────────────────────
LOG_DIR  = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = ROOT / "data" / "raw" / "p8_crossleague_transfers.parquet"
PSS_PATH = ROOT / "data" / "processed" / "player_season_stats.parquet"

# ─── 로깅 ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "p8_crossleague.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("p8_cross")

# ─── 수집 대상 리그 (FBref 직접 확인 기반) ───────────────────────────────────
# comp ID: FBref 내부 리그 식별자
# slug: URL 내 리그명 세그먼트
FOREIGN_LEAGUES = [
    {"name": "Bundesliga", "comp": 20, "slug": "Bundesliga",  "level_diff": 0},
    {"name": "La Liga",    "comp": 12, "slug": "La-Liga",     "level_diff": 0},
    {"name": "Serie A",    "comp": 11, "slug": "Serie-A",     "level_diff": 0},
    {"name": "Ligue 1",    "comp": 13, "slug": "Ligue-1",     "level_diff": 0},
]

# 수집 시즌: 2009-10 ~ 2023-24
# FBref URL 형식: "2009-2010"  /  EPL parquet 시즌: "2010/11"
SEASONS = [
    {"fbref_id": f"{y}-{y+1}", "foreign_season": f"{y}/{str(y+1)[-2:]}",
     "epl_season": f"{y+1}/{str(y+2)[-2:]}"}
    for y in range(2009, 2024)
]

FBREF_BASE = "https://fbref.com"


# ─── 유틸 ────────────────────────────────────────────────────────────────────
def norm(name: str) -> str:
    """악센트 제거 + 소문자 + 공백 정리."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def to_f(v) -> float:
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def pos_code(pos: str) -> float:
    p = (pos or "").upper()
    if "GK" in p:          return 0.0
    if "DF" in p and "MF" not in p and "FW" not in p: return 1.0
    if "FW" in p:          return 3.0
    return 2.0


def age_bucket(age) -> float:
    try:
        a = float(str(age).split("-")[0])
    except (TypeError, ValueError):
        return 2.0
    if a < 21:   return 1.0
    if a <= 25:  return 2.0
    if a <= 30:  return 3.0
    return 4.0


# ─── FBref 리그 시즌 통계 파싱 (Bundesliga 2018-19 구조 확인 기반) ─────────
def parse_foreign_stats(soup: BeautifulSoup) -> pd.DataFrame:
    """FBref 해외 리그 시즌 통계 페이지 → 선수 DataFrame.

    확인된 data-stat 속성:
      player, team, position, age, games, games_starts,
      minutes, minutes_90s, goals, assists, goals_assists
    나이 형식: "32" (Championship과 달리 숫자만, 대시 없음)
    """
    all_html = str(soup)
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        all_html += str(c)
    soup2 = BeautifulSoup(all_html, "html.parser")

    table = soup2.find("table", {"id": "stats_standard"})
    if table is None:
        return pd.DataFrame()

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cls = tr.get("class", [])
        if "thead" in cls or "spacer" in cls:
            continue

        def cell(stat):
            td = tr.find(["td", "th"], {"data-stat": stat})
            return td.get_text(strip=True) if td else ""

        player = cell("player")
        if not player or player == "Player":
            continue

        mp   = to_f(cell("games"))
        strt = to_f(cell("games_starts"))
        mins = to_f(cell("minutes"))
        s90  = to_f(cell("minutes_90s"))
        gls  = to_f(cell("goals"))
        ast  = to_f(cell("assists"))
        # 나이: Bundesliga는 "32" 형식 (Championship "32-143"과 다름 — 확인됨)
        age_raw = cell("age")
        try:
            age_val = float(age_raw.split("-")[0])
        except (ValueError, AttributeError):
            age_val = 0.0

        denom   = s90 if s90 > 0 else 1.0
        rows.append({
            "player":    player,
            "player_norm": norm(player),
            "team_old":  cell("team"),
            "pos":       cell("position"),
            "age":       age_val,
            "mp_old":    mp,
            "starts_old": strt,
            "min_old":   mins,
            "90s_old":   s90,
            "gls_p90":   round(gls / denom, 4),
            "ast_p90":   round(ast / denom, 4),
            "ga_p90":    round((gls + ast) / denom, 4),
        })

    return pd.DataFrame(rows)


# ─── 이적 레코드 생성 ────────────────────────────────────────────────────────
def build_records(
    foreign_df: pd.DataFrame,
    epl_df: pd.DataFrame,
    pss_all: pd.DataFrame,
    foreign_season: str,
    epl_season: str,
    league_name: str,
    level_diff: int,
) -> list:
    """외국 리그 스탯 × EPL 다음 시즌 스탯 매칭 → 이적 레코드."""
    if foreign_df.empty or epl_df.empty:
        return []

    epl_df = epl_df.copy()
    epl_df["player_norm"] = epl_df["player"].map(norm)
    epl_lookup = {row["player_norm"]: row for _, row in epl_df.iterrows()}

    # 이전 시즌 EPL 재직 여부 확인 (재입성 vs 신규 입성 구분용)
    prev_epl_season_year = int(epl_season.split("/")[0]) - 1
    prev_epl_season = f"{prev_epl_season_year}/{str(prev_epl_season_year+1)[-2:]}"
    prev_epl_players = set(
        pss_all[pss_all["season"] == prev_epl_season]["player"].map(norm).dropna()
    )

    records = []
    for _, fr in foreign_df.iterrows():
        pn = fr["player_norm"]
        if pn not in epl_lookup:
            continue
        if fr["min_old"] < 450:     # 외국 리그 최소 출전
            continue

        er = epl_lookup[pn]
        team_new = er["team"]
        team_old_canon = fr["team_old"]

        # 핵심: 클럽이 실제로 바뀌어야 함
        # (Championship과 달리 거의 항상 바뀜)
        # team_old는 외국 리그 팀명, team_new는 EPL 팀명 → 항상 다름

        adapted    = 1 if to_f(er.get("min", 0)) >= 900 else 0
        was_start  = 1.0 if (fr["mp_old"] > 0 and
                              fr["starts_old"] / fr["mp_old"] >= 0.6) else 0.0

        # EPL 경험 (이전 EPL 시즌 존재 여부)
        epl_exp = 1 if pn in prev_epl_players else 0

        records.append({
            # 식별자
            "player":        fr["player"],
            "season_old":    foreign_season,
            "season_new":    epl_season,
            "team_old":      team_old_canon,
            "team_new":      team_new,
            "source_league": league_name,
            # 외국 리그 스탯
            "g_a_per90_old":  fr["ga_p90"],
            "gls_per90_old":  fr["gls_p90"],
            "ast_per90_old":  fr["ast_p90"],
            "min_old":        fr["min_old"],
            "90s_old":        fr["90s_old"],
            "mp_old":         fr["mp_old"],
            "starts_old":     fr["starts_old"],
            # EPL 스탯
            "g_a_per90_new":  to_f(er.get("g_a_1", 0)),
            "gls_per90_new":  to_f(er.get("gls_1", 0)),
            "ast_per90_new":  to_f(er.get("ast_1", 0)),
            "min_new":        to_f(er.get("min", 0)),
            "90s_new":        to_f(er.get("90s", 0)),
            "mp_new":         to_f(er.get("mp", 0)),
            "starts_new":     to_f(er.get("starts", 0)),
            # 속성
            "age":            fr["age"],
            "pos":            fr["pos"],
            "position":       fr["pos"],
            "adapted":        adapted,
            # 파생 피처
            "pos_code":           pos_code(fr["pos"]),
            "age_bucket":         age_bucket(fr["age"]),
            "was_starter":        was_start,
            "moving_up":          1,        # 해외→EPL = 리그 레벨 이동
            "epl_experience":     epl_exp,
            "transfer_count":     0,
            "league_level_diff":  level_diff,
            # NaN 허용 피처 (train.py fillna(0) 처리)
            "market_value":       None,
            "market_value_old":   None,
            "height_cm":          None,
            "elo_old":            None,
            "elo_new":            None,
            "elo_diff":           None,
            "points_diff":        None,
            "style_distance":     None,
            "style_match_pct":    None,
            "hist_adapt_rate":    None,
            "mv_vs_squad":        None,
            "squad_avg_mv":       None,
            "new_team_points":    None,
            "old_team_points":    None,
        })

    return records


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("🌍 P8 해외→EPL 크로스리그 이적 데이터 수집 시작")
    log.info(f"   대상: {[l['name'] for l in FOREIGN_LEAGUES]}")
    log.info(f"   시즌: {SEASONS[0]['fbref_id']} ~ {SEASONS[-1]['fbref_id']}")
    log.info(f"   총 페이지: {len(FOREIGN_LEAGUES) * len(SEASONS)}개")
    log.info("=" * 60)

    pss_all = pd.read_parquet(PSS_PATH)
    log.info(f"EPL 스탯 로드: {len(pss_all)}행")

    crawler = BaseCrawlerAgent(source_name="p8_cross", min_interval=6.0)
    all_records = []

    # 이전 결과 이어받기
    if OUT_PATH.exists():
        prev = pd.read_parquet(OUT_PATH)
        all_records = prev.to_dict("records")
        log.info(f"이전 결과 로드: {len(all_records)}건")

    try:
        for league in FOREIGN_LEAGUES:
            log.info(f"\n{'─'*50}")
            log.info(f"🏟️  {league['name']} (comp {league['comp']})")

            for seas in SEASONS:
                task_key = f"{league['name']}_{seas['fbref_id']}"

                if crawler.is_completed(task_key):
                    log.info(f"  ⏭️  {seas['fbref_id']} 스킵")
                    continue

                url = (
                    f"{FBREF_BASE}/en/comps/{league['comp']}/"
                    f"{seas['fbref_id']}/stats/"
                    f"{seas['fbref_id']}-{league['slug']}-Stats"
                )
                log.info(f"  📥 {seas['fbref_id']} → EPL {seas['epl_season']}")

                soup = crawler.fetch(url)
                if soup is None:
                    log.warning(f"  ❌ 로드 실패 — 스킵")
                    crawler.mark_failed(task_key, "fetch failed")
                    continue

                foreign_df = parse_foreign_stats(soup)
                if foreign_df.empty:
                    log.warning(f"  ⚠️  파싱 결과 없음")
                    crawler.mark_failed(task_key, "parse empty")
                    continue

                log.info(f"     {league['name']} 선수 {len(foreign_df)}명 파싱")

                # 해당 EPL 시즌 데이터
                epl_df = pss_all[pss_all["season"] == seas["epl_season"]].copy()
                if epl_df.empty:
                    log.warning(f"  ⚠️  EPL {seas['epl_season']} 데이터 없음")
                    crawler.mark_completed(task_key)
                    continue

                records = build_records(
                    foreign_df, epl_df, pss_all,
                    seas["foreign_season"], seas["epl_season"],
                    league["name"], league["level_diff"],
                )
                log.info(f"     이적 매칭: {len(records)}건")

                all_records.extend(records)
                crawler.mark_completed(task_key)

                # 중간 저장
                if all_records:
                    pd.DataFrame(all_records).to_parquet(
                        OUT_PATH, index=False, engine="pyarrow"
                    )

    finally:
        crawler.close()

    # 최종 저장
    log.info(f"\n{'='*60}")
    if not all_records:
        log.error("❌ 수집 레코드 없음")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    before = len(df)
    df = df.drop_duplicates(subset=["player", "season_old", "season_new"])
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")

    log.info(f"✅ 저장: {OUT_PATH.relative_to(ROOT)}")
    log.info(f"   총 {len(df)}건 (중복 제거 {before-len(df)}건)")
    log.info(f"   리그별 분포:")
    for lg, n in df["source_league"].value_counts().items():
        log.info(f"     {lg}: {n}건")
    log.info(f"   adapted=1: {int(df['adapted'].sum())}건 "
             f"({df['adapted'].mean()*100:.1f}%)")
    log.info(f"\n다음: python scripts/p8_merge_crossleague.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
