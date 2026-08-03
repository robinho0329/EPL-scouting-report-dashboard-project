"""scripts/p8_championship_crawler.py — Championship → EPL 승격 이적 데이터 수집.

P8 이적 적응 모델 학습 데이터 확장 목적:
  - 기존: EPL 내부 이적 852건 (실사용 173건)
  - 목표: Championship→EPL 승격 케이스 추가 → train 650+

수집 대상: 2003-04 ~ 2023-24 Championship 승격팀 (21시즌 × 3팀 = 63팀)
          EPL 데이터 보유 범위(2004/05~2024/25) 전체 커버
데이터 출처: FBref (undetected-chromedriver로 Cloudflare 우회, 레이트 리밋 6초)

실행 방법:
  저장소 루트에서 (가상환경 활성화 후):
    python scripts/p8_championship_crawler.py

  ※ Chrome 브라우저 창이 자동으로 열립니다 (닫지 마세요)
  ※ 중단했다가 재실행하면 체크포인트에서 이어집니다

소요 시간: 약 40~60분
결과 파일: data/raw/p8_championship_transfers.parquet
다음 단계: python scripts/p8_merge_external_data.py
"""

import sys
import logging
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment

# 프로젝트 루트를 sys.path에 추가 (crawlers 임포트 가능하도록)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crawlers.base_agent import BaseCrawlerAgent  # Selenium + Cloudflare 우회

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH   = ROOT / "data" / "raw" / "p8_championship_transfers.parquet"
PSS_PATH   = ROOT / "data" / "processed" / "player_season_stats.parquet"

# ─── 로깅 ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "p8_crawl.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("p8_crawl")

# ─── 승격팀 목록 ──────────────────────────────────────────────────────────────
# Championship 시즌(FBref URL) → EPL 진입 시즌(parquet) + FBref 팀명
# 2003-04부터 2023-24까지 전 시즌 (FBref comp 10 커버 범위)
PROMOTED = {
    # ── 2000년대 초반 ─────────────────────────────────────────────
    # ※ 팀명은 FBref stats 페이지의 실제 Squad 표기 기준 (공식명 아님)
    "2003-2004": {
        "champ_season": "2003/04",
        "epl_season":   "2004/05",
        "teams": ["Norwich City", "West Brom", "Crystal Palace"],
    },
    "2004-2005": {
        "champ_season": "2004/05",
        "epl_season":   "2005/06",
        "teams": ["Sunderland", "Wigan Athletic", "West Ham United"],
    },
    "2005-2006": {
        "champ_season": "2005/06",
        "epl_season":   "2006/07",
        "teams": ["Reading", "Sheffield United", "Watford"],
    },
    "2006-2007": {
        "champ_season": "2006/07",
        "epl_season":   "2007/08",
        "teams": ["Sunderland", "Birmingham City", "Derby County"],
    },
    "2007-2008": {
        "champ_season": "2007/08",
        "epl_season":   "2008/09",
        "teams": ["West Brom", "Stoke City", "Hull City"],
    },
    "2008-2009": {
        "champ_season": "2008/09",
        "epl_season":   "2009/10",
        "teams": ["Wolves", "Birmingham City", "Burnley"],
    },
    "2009-2010": {
        "champ_season": "2009/10",
        "epl_season":   "2010/11",
        "teams": ["Newcastle United", "West Brom", "Blackpool"],
    },
    "2010-2011": {
        "champ_season": "2010/11",
        "epl_season":   "2011/12",
        "teams": ["QPR", "Norwich City", "Swansea City"],
    },
    "2011-2012": {
        "champ_season": "2011/12",
        "epl_season":   "2012/13",
        "teams": ["Reading", "Southampton", "West Ham United"],
    },
    "2012-2013": {
        "champ_season": "2012/13",
        "epl_season":   "2013/14",
        "teams": ["Cardiff City", "Hull City", "Crystal Palace"],
    },
    "2013-2014": {
        "champ_season": "2013/14",
        "epl_season":   "2014/15",
        "teams": ["Leicester City", "Burnley", "QPR"],
    },
    "2014-2015": {
        "champ_season": "2014/15",
        "epl_season":   "2015/16",
        "teams": ["Bournemouth", "Watford", "Norwich City"],
    },
    "2015-2016": {
        "champ_season": "2015/16",
        "epl_season":   "2016/17",
        "teams": ["Burnley", "Middlesbrough", "Hull City"],
    },
    "2016-2017": {
        "champ_season": "2016/17",
        "epl_season":   "2017/18",
        "teams": ["Newcastle United", "Brighton", "Huddersfield Town"],
    },
    "2017-2018": {
        "champ_season": "2017/18",
        "epl_season":   "2018/19",
        "teams": ["Wolves", "Cardiff City", "Fulham"],
    },
    "2018-2019": {
        "champ_season": "2018/19",
        "epl_season":   "2019/20",
        "teams": ["Norwich City", "Sheffield United", "Aston Villa"],
    },
    # ── 기존 5시즌 재수집 (체크포인트 삭제 후 재실행 필요) ───────────
    "2019-2020": {
        "champ_season": "2019/20",
        "epl_season":   "2020/21",
        "teams": ["Leeds United", "West Brom", "Fulham"],
    },
    "2020-2021": {
        "champ_season": "2020/21",
        "epl_season":   "2021/22",
        "teams": ["Norwich City", "Watford", "Brentford"],
    },
    "2021-2022": {
        "champ_season": "2021/22",
        "epl_season":   "2022/23",
        "teams": ["Fulham", "Bournemouth", "Nottingham Forest"],
    },
    "2022-2023": {
        "champ_season": "2022/23",
        "epl_season":   "2023/24",
        "teams": ["Burnley", "Sheffield United", "Luton Town"],
    },
    "2023-2024": {
        "champ_season": "2023/24",
        "epl_season":   "2024/25",
        "teams": ["Leicester City", "Ipswich Town", "Southampton"],
    },
}

# FBref팀명 → EPL parquet 캐노니컬 팀명 (config/team_mapping.json 기준)
FBREF_TO_CANONICAL = {
    # 2019-24 (기존)
    "Leeds United":             "Leeds",
    "West Bromwich Albion":     "West Brom",
    "Fulham":                   "Fulham",
    "Norwich City":             "Norwich",
    "Watford":                  "Watford",
    "Brentford":                "Brentford",
    "AFC Bournemouth":          "Bournemouth",
    "Nottingham Forest":        "Nott'm Forest",
    "Burnley":                  "Burnley",
    "Sheffield United":         "Sheffield United",
    "Luton Town":               "Luton",
    "Leicester City":           "Leicester",
    "Ipswich Town":             "Ipswich",
    "Southampton":              "Southampton",
    # 2003-19 (FBref 실제 표기 기준)
    "Crystal Palace":       "Crystal Palace",
    "Sunderland":           "Sunderland",
    "Wigan Athletic":       "Wigan",
    "West Ham":             "West Ham",
    "West Ham United":      "West Ham",       # 폴백
    "Reading":              "Reading",
    "Birmingham City":      "Birmingham",
    "Derby County":         "Derby",
    "Stoke City":           "Stoke",
    "Hull City":            "Hull",
    "Wolves":               "Wolves",
    "Blackpool":            "Blackpool",
    "QPR":                  "QPR",            # FBref 실제 표기 (검증완료)
    "Queens Park Rgrs":     "QPR",
    "Queens Park Rangers":  "QPR",
    "Swansea City":         "Swansea",
    "Cardiff City":         "Cardiff",
    "Newcastle United":     "Newcastle",      # FBref 실제 표기 (검증완료)
    "Newcastle Utd":        "Newcastle",
    "Brighton":             "Brighton",
    "Brighton & Hove Albion": "Brighton",     # 폴백
    "Huddersfield Town":    "Huddersfield",   # FBref 실제 표기 (검증완료)
    "Huddersfield":         "Huddersfield",
    "Middlesbrough":        "Middlesbrough",
    "Aston Villa":          "Aston Villa",
    "West Brom":            "West Brom",
    "Bournemouth":          "Bournemouth",
    "Nott'm Forest":        "Nott'm Forest",
    "Nottingham Forest":    "Nott'm Forest",  # FBref 실제 표기 (검증완료)
    "Nottm Forest":         "Nott'm Forest",
}

FBREF_BASE = "https://fbref.com"


# ─── 유틸 ────────────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    """악센트 제거 + 소문자 + 공백 정리 (선수명 매칭용)."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def pos_to_code(pos: str) -> float:
    """FBref pos → 숫자 코드 (GK=0, DF=1, MF=2, FW=3)."""
    if not isinstance(pos, str):
        return 2.0
    p = pos.upper()
    if "GK" in p:
        return 0.0
    if "DF" in p and "MF" not in p and "FW" not in p:
        return 1.0
    if "FW" in p:
        return 3.0
    return 2.0


def age_to_bucket(age) -> float:
    """나이 → 버킷 (1=<21, 2=21-25, 3=26-30, 4=>30)."""
    try:
        a = float(str(age).split("-")[0])
    except (TypeError, ValueError):
        return 2.0
    if a < 21:
        return 1.0
    if a <= 25:
        return 2.0
    if a <= 30:
        return 3.0
    return 4.0


def to_float(v) -> float:
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


# ─── FBref Championship 스탯 파싱 ────────────────────────────────────────────
def parse_champ_stats(soup: BeautifulSoup, fbref_teams: list) -> pd.DataFrame:
    """Championship 시즌 스탯 페이지 → 승격팀 선수 DataFrame.

    FBref는 일부 테이블을 HTML 주석 안에 넣으므로 주석도 탐색.
    """
    # 주석 포함 전체 HTML 재파싱
    all_html = str(soup)
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        all_html += str(comment)
    combined = BeautifulSoup(all_html, "html.parser")

    table = combined.find("table", {"id": "stats_standard"})
    if table is None:
        # ID 없으면 가장 큰 테이블로 폴백
        tables = combined.find_all("table")
        if not tables:
            log.warning("  ⚠️  테이블을 전혀 찾지 못했습니다")
            return pd.DataFrame()
        table = max(tables, key=lambda t: len(t.find_all("tr")))
        log.warning("  ⚠️  stats_standard 없음 → 최대 테이블로 폴백")

    tbody = table.find("tbody")
    if tbody is None:
        log.warning("  ⚠️  tbody 없음")
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        cls = tr.get("class", [])
        if "thead" in cls or "spacer" in cls:
            continue

        def cell(stat: str) -> str:
            td = tr.find(["td", "th"], {"data-stat": stat})
            return td.get_text(strip=True) if td else ""

        squad = cell("team")
        if not squad or squad not in fbref_teams:
            continue

        player = cell("player")
        if not player or player in ("Player", ""):
            continue

        age_raw = cell("age")  # 형식: "25-143" (연-일)
        age_val = float(age_raw.split("-")[0]) if age_raw and "-" in age_raw else to_float(age_raw)

        mp     = to_float(cell("games"))
        starts = to_float(cell("games_starts"))
        mins   = to_float(cell("minutes"))
        s90    = to_float(cell("minutes_90s"))
        gls    = to_float(cell("goals"))
        ast    = to_float(cell("assists"))
        pos    = cell("position")

        denom   = s90 if s90 > 0 else 1.0
        gls_p90 = round(gls / denom, 4)
        ast_p90 = round(ast / denom, 4)
        ga_p90  = round((gls + ast) / denom, 4)

        rows.append({
            "player":      player,
            "squad_fbref": squad,
            "pos":         pos,
            "age":         age_val,
            "mp":          mp,
            "starts":      starts,
            "min":         mins,
            "90s":         s90,
            "gls":         gls,
            "ast":         ast,
            "gls_p90":     gls_p90,
            "ast_p90":     ast_p90,
            "ga_p90":      ga_p90,
        })

    df = pd.DataFrame(rows)
    if len(df):
        log.info(f"  → 승격팀 선수 {len(df)}명 | 팀: {df['squad_fbref'].unique().tolist()}")
    else:
        log.warning("  ⚠️  승격팀 선수 데이터 없음")
    return df


# ─── 이적 레코드 생성 ────────────────────────────────────────────────────────
def build_records(
    champ_df: pd.DataFrame,
    epl_df: pd.DataFrame,
    champ_season: str,
    epl_season: str,
) -> list:
    if champ_df.empty or epl_df.empty:
        return []

    champ_df = champ_df.copy()
    champ_df["_norm"] = champ_df["player"].map(normalize_name)

    epl_df = epl_df.copy()
    epl_df["_norm"] = epl_df["player"].map(normalize_name)
    epl_lookup = {row["_norm"]: row for _, row in epl_df.iterrows()}

    records = []
    for _, crow in champ_df.iterrows():
        norm = crow["_norm"]
        if norm not in epl_lookup:
            continue

        # Championship 최소 출전 필터 (450분 = 5경기 풀타임)
        if crow["min"] < 450:
            continue

        erow = epl_lookup[norm]
        team_old = FBREF_TO_CANONICAL.get(crow["squad_fbref"], crow["squad_fbref"])
        team_new = erow["team"]

        adapted    = 1 if to_float(erow.get("min", 0)) >= 900 else 0
        was_starter = 1.0 if (crow["mp"] > 0 and crow["starts"] / crow["mp"] >= 0.6) else 0.0

        records.append({
            # ── 식별자 ──
            "player":        crow["player"],
            "season_old":    champ_season,
            "season_new":    epl_season,
            "team_old":      team_old,
            "team_new":      team_new,
            # ── Championship 스탯 ──
            "g_a_per90_old":  crow["ga_p90"],
            "gls_per90_old":  crow["gls_p90"],
            "ast_per90_old":  crow["ast_p90"],
            "min_old":        crow["min"],
            "90s_old":        crow["90s"],
            "mp_old":         crow["mp"],
            "starts_old":     crow["starts"],
            # ── EPL 스탯 ──
            "g_a_per90_new":  to_float(erow.get("g_a_1", 0)),
            "gls_per90_new":  to_float(erow.get("gls_1", 0)),
            "ast_per90_new":  to_float(erow.get("ast_1", 0)),
            "min_new":        to_float(erow.get("min", 0)),
            "90s_new":        to_float(erow.get("90s", 0)),
            "mp_new":         to_float(erow.get("mp", 0)),
            "starts_new":     to_float(erow.get("starts", 0)),
            # ── 속성 ──
            "age":            crow["age"],
            "pos":            crow["pos"],
            "position":       crow["pos"],
            "adapted":        adapted,
            # ── 파생 피처 ──
            "pos_code":            pos_to_code(crow["pos"]),
            "age_bucket":          age_to_bucket(crow["age"]),
            "was_starter":         was_starter,
            "moving_up":           1,
            "epl_experience":      0,
            "transfer_count":      0,
            "league_level_diff":   1,   # Championship→EPL 리그 레벨 차이 (새 피처)
            # ── 결측 허용 (train.py에서 fillna(0) 처리) ──
            "market_value":     None,
            "market_value_old": None,
            "height_cm":        None,
            "elo_old":          None,
            "elo_new":          None,
            "elo_diff":         None,
            "points_diff":      None,
            "style_distance":   None,
            "style_match_pct":  None,
            "hist_adapt_rate":  None,
            "mv_vs_squad":      None,
            "squad_avg_mv":     None,
            "new_team_points":  None,
            "old_team_points":  None,
        })

    log.info(f"  → 매칭 성공: {len(records)}건 / 승격팀 {len(champ_df)}명 중")
    return records


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("🏴󠁧󠁢󠁥󠁮󠁧󠁿 P8 Championship→EPL 외부 데이터 수집 시작")
    log.info("   Chrome 창이 열립니다 — 닫지 마세요!")
    log.info("=" * 60)

    # EPL 전체 스탯 로드
    log.info(f"📂 EPL 스탯 로드...")
    pss_all = pd.read_parquet(PSS_PATH)
    log.info(f"   {pss_all.shape[0]}행, {pss_all['season'].min()}~{pss_all['season'].max()}")

    # 크롤러 초기화 (체크포인트: crawl_checkpoint.db의 "p8_champ" 소스)
    crawler = BaseCrawlerAgent(source_name="p8_champ", min_interval=6.0)

    all_records = []

    # 이전 결과가 있으면 로드 (재실행 시 이어서)
    if OUT_PATH.exists():
        df_prev = pd.read_parquet(OUT_PATH)
        all_records = df_prev.to_dict("records")
        log.info(f"📌 이전 결과 로드: {len(all_records)}건")

    try:
        for season_id, info in PROMOTED.items():
            task_key     = f"season_{season_id}"
            champ_season = info["champ_season"]
            epl_season   = info["epl_season"]
            fbref_teams  = info["teams"]

            # 이미 완료된 시즌 스킵
            if crawler.is_completed(task_key):
                log.info(f"\n⏭️  {season_id} 이미 완료 — 스킵")
                continue

            log.info(f"\n{'─'*50}")
            log.info(f"📅 {season_id} Championship")
            log.info(f"   승격팀: {fbref_teams}")
            log.info(f"   {champ_season} → EPL {epl_season}")

            url = (
                f"{FBREF_BASE}/en/comps/10/{season_id}/stats/"
                f"{season_id}-Championship-Stats"
            )

            soup = crawler.fetch(url)
            if soup is None:
                log.error(f"  ❌ 페이지 로드 실패 — 스킵")
                crawler.mark_failed(task_key, "page load failed")
                continue

            champ_df = parse_champ_stats(soup, fbref_teams)
            if champ_df.empty:
                log.warning(f"  ⚠️  파싱 결과 없음 — 스킵")
                crawler.mark_failed(task_key, "parse empty")
                continue

            # EPL 데이터: 해당 시즌 + 승격팀
            canonical_teams = [FBREF_TO_CANONICAL.get(t, t) for t in fbref_teams]
            epl_df = pss_all[
                (pss_all["season"] == epl_season) &
                (pss_all["team"].isin(canonical_teams))
            ].copy()
            log.info(f"  📊 EPL {epl_season} 승격팀 선수: {len(epl_df)}명")

            records = build_records(champ_df, epl_df, champ_season, epl_season)
            all_records.extend(records)

            # 중간 저장 (체크포인트 역할)
            if all_records:
                pd.DataFrame(all_records).to_parquet(OUT_PATH, index=False, engine="pyarrow")
                log.info(f"  💾 중간 저장 완료 (누적 {len(all_records)}건)")

            crawler.mark_completed(task_key)

    finally:
        crawler.close()

    # 최종 처리
    log.info(f"\n{'='*60}")
    if not all_records:
        log.error("❌ 수집된 레코드가 없습니다.")
        sys.exit(1)

    df_out = pd.DataFrame(all_records)
    before = len(df_out)
    df_out = df_out.drop_duplicates(subset=["player", "season_old", "season_new"])
    df_out.to_parquet(OUT_PATH, index=False, engine="pyarrow")

    log.info(f"✅ 최종 저장: {OUT_PATH.relative_to(ROOT)}")
    log.info(f"   총 {len(df_out)}건 (중복 제거: {before - len(df_out)}건)")
    log.info(f"   적응 성공(adapted=1): {int(df_out['adapted'].sum())}건")
    log.info(f"\n   시즌별 분포:")
    for s, n in df_out["season_old"].value_counts().sort_index().items():
        log.info(f"     {s}: {n}명")

    log.info(f"\n{'='*60}")
    log.info("✅ 크롤링 완료!")
    log.info("다음 단계: python scripts/p8_merge_external_data.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
