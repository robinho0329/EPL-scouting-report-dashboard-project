"""scripts/add_odds_features.py — Football-Data.co.uk EPL 오즈 피처 추가

Football-Data.co.uk에서 EPL 시즌별 오즈 CSV를 다운로드해
match_features.parquet에 아래 피처를 추가합니다:

추가 피처:
  b365_home_prob  : Bet365 홈 승 내재 확률 (1/B365H 정규화)
  b365_draw_prob  : Bet365 무승부 내재 확률
  b365_away_prob  : Bet365 어웨이 승 내재 확률
  market_home_fav : 시장이 홈 팀 선호 여부 (b365_home_prob > b365_away_prob)
  draw_market_prob: 무승부 시장 확률 (무승부 예측 핵심 피처)
  odds_overround  : 북메이커 마진 (1 + 오버라운드 → 1에 가까울수록 효율적 시장)

사용법:
  python scripts/add_odds_features.py

의존성:
  pip install requests pandas pyarrow
"""

import io
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("add_odds_features")

ROOT           = Path(__file__).resolve().parent.parent
DATA_FEAT      = ROOT / "data" / "features"
MATCH_FEAT_OUT = DATA_FEAT / "match_features.parquet"
ODDS_CACHE_DIR = ROOT / "data" / "raw" / "odds_cache"
ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_DIR = ROOT / "config"

# ─────────────────────────────────────────────────────────────────────────────
# Football-Data.co.uk EPL 시즌 URL 패턴
# E0 = EPL (잉글랜드 1부리그)
# 시즌 코드: 0001 (00/01) ~ 2425 (24/25)
# ─────────────────────────────────────────────────────────────────────────────
FD_BASE = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# 수집할 시즌 목록 (2000/01 ~ 2024/25, Football-Data.co.uk에서 제공하는 범위)
SEASONS_TO_FETCH = [
    "0001", "0102", "0203", "0304", "0405", "0506",
    "0607", "0708", "0809", "0910", "1011", "1112",
    "1213", "1314", "1415", "1516", "1617", "1718",
    "1819", "1920", "2021", "2122", "2223", "2324", "2425",
]

# Football-Data.co.uk 팀 이름 → 우리 표준 이름 매핑
FD_TEAM_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Birmingham": "Birmingham",
    "Blackburn": "Blackburn",
    "Blackpool": "Blackpool",
    "Bolton": "Bolton",
    "Bournemouth": "Bournemouth",
    "Bradford": "Bradford",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff",
    "Charlton": "Charlton",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Derby": "Derby",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Man City",
    "Man United": "Man United",
    "Middlesbrough": "Middlesbrough",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "Nottm Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Oldham": "Oldham",
    "Portsmouth": "Portsmouth",
    "QPR": "QPR",
    "Reading": "Reading",
    "Sheffield United": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Southampton": "Southampton",
    "Stoke": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Tottenham": "Spurs",
    "Watford": "Watford",
    "West Brom": "West Brom",
    "West Ham": "West Ham",
    "Wigan": "Wigan",
    "Wimbledon": "Wimbledon",
    "Wolves": "Wolves",
}


def fd_season_to_our_season(fd_season: str) -> str:
    """'0102' → '2001/02'"""
    yy1 = int(fd_season[:2])
    yy2 = int(fd_season[2:])
    year1 = 2000 + yy1 if yy1 >= 0 else 1900 + yy1
    return f"{year1}/{fd_season[2:]}"


def download_season(season_code: str) -> pd.DataFrame:
    """단일 시즌 오즈 CSV 다운로드 (캐시 우선)."""
    cache_path = ODDS_CACHE_DIR / f"E0_{season_code}.csv"

    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, encoding="latin-1", on_bad_lines="skip")
            if len(df) > 0:
                logger.info(f"    캐시 사용: {cache_path.name} ({len(df)}행)")
                return df
        except Exception:
            pass

    url = FD_BASE.format(season=season_code)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.football-data.co.uk/englandm.php",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="latin-1", on_bad_lines="skip")
        if len(df) > 0:
            df.to_csv(cache_path, index=False, encoding="utf-8")
            logger.info(f"    다운로드: {season_code} ({len(df)}행)")
            return df
        else:
            logger.warning(f"    빈 데이터: {season_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"    다운로드 실패: {season_code} — {e}")
        return pd.DataFrame()


def parse_odds_df(df: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """오즈 CSV를 표준화된 형식으로 변환."""
    if df.empty:
        return pd.DataFrame()

    required_cols = ["Date", "HomeTeam", "AwayTeam"]
    odds_cols_b365 = ["B365H", "B365D", "B365A"]
    odds_cols_alt = [["IWH", "IWD", "IWA"], ["PSH", "PSD", "PSA"], ["WHH", "WHD", "WHA"]]

    if not all(c in df.columns for c in required_cols):
        return pd.DataFrame()

    # Bet365 오즈 사용 (없으면 대안 오즈 사용)
    h_col, d_col, a_col = None, None, None
    if all(c in df.columns for c in odds_cols_b365):
        h_col, d_col, a_col = "B365H", "B365D", "B365A"
    else:
        for alt in odds_cols_alt:
            if all(c in df.columns for c in alt):
                h_col, d_col, a_col = alt
                break

    if not h_col:
        logger.warning(f"    오즈 컬럼 없음: {season_code}")
        return pd.DataFrame()

    result = df[["Date", "HomeTeam", "AwayTeam", h_col, d_col, a_col]].copy()
    result = result.rename(columns={h_col: "odds_H", d_col: "odds_D", a_col: "odds_A"})
    result = result.dropna(subset=["odds_H", "odds_D", "odds_A"])

    # 날짜 파싱 (DD/MM/YY 또는 DD/MM/YYYY)
    result["MatchDate"] = pd.to_datetime(result["Date"], dayfirst=True, errors="coerce")
    result = result.dropna(subset=["MatchDate"])

    # 팀 이름 표준화
    result["HomeTeam"] = result["HomeTeam"].map(FD_TEAM_MAP).fillna(result["HomeTeam"])
    result["AwayTeam"] = result["AwayTeam"].map(FD_TEAM_MAP).fillna(result["AwayTeam"])

    # 내재 확률 계산 (정규화)
    result["odds_H"] = pd.to_numeric(result["odds_H"], errors="coerce")
    result["odds_D"] = pd.to_numeric(result["odds_D"], errors="coerce")
    result["odds_A"] = pd.to_numeric(result["odds_A"], errors="coerce")
    result = result.dropna(subset=["odds_H", "odds_D", "odds_A"])

    # 오버라운드 제거 후 정규화
    inv_H = 1.0 / result["odds_H"].clip(1.01)
    inv_D = 1.0 / result["odds_D"].clip(1.01)
    inv_A = 1.0 / result["odds_A"].clip(1.01)
    total = inv_H + inv_D + inv_A

    result["b365_home_prob"]  = (inv_H / total).round(4)
    result["b365_draw_prob"]  = (inv_D / total).round(4)
    result["b365_away_prob"]  = (inv_A / total).round(4)
    result["odds_overround"]  = (total - 1.0).round(4)
    result["draw_market_prob"] = result["b365_draw_prob"]  # 별칭 (가독성)
    result["market_home_fav"] = (result["b365_home_prob"] > result["b365_away_prob"]).astype(int)

    return result[["MatchDate", "HomeTeam", "AwayTeam",
                   "b365_home_prob", "b365_draw_prob", "b365_away_prob",
                   "draw_market_prob", "market_home_fav", "odds_overround"]]


def build_odds_master() -> pd.DataFrame:
    """모든 시즌 오즈 데이터 통합."""
    logger.info("Football-Data.co.uk 오즈 데이터 수집 시작")
    frames = []
    for sc in SEASONS_TO_FETCH:
        raw = download_season(sc)
        parsed = parse_odds_df(raw, sc)
        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        logger.error("다운로드된 오즈 데이터 없음")
        return pd.DataFrame()

    master = pd.concat(frames, ignore_index=True)
    master = master.drop_duplicates(subset=["MatchDate", "HomeTeam", "AwayTeam"])
    logger.info(f"오즈 마스터 데이터: {len(master)}행 ({master['MatchDate'].dt.year.min()}-{master['MatchDate'].dt.year.max()})")
    return master


def merge_odds_into_features(odds_df: pd.DataFrame) -> None:
    """match_features.parquet에 오즈 피처 조인 후 덮어쓰기."""
    if not MATCH_FEAT_OUT.exists():
        logger.error(f"match_features.parquet 없음: {MATCH_FEAT_OUT}")
        return

    logger.info("match_features.parquet 로드 중...")
    match_df = pd.read_parquet(MATCH_FEAT_OUT)
    logger.info(f"  현재 match_features: {match_df.shape}")

    odds_cols = ["b365_home_prob", "b365_draw_prob", "b365_away_prob",
                 "draw_market_prob", "market_home_fav", "odds_overround"]

    # 기존에 오즈 피처가 있으면 제거 후 재조인 (갱신)
    existing_odds = [c for c in odds_cols if c in match_df.columns]
    if existing_odds:
        logger.info(f"  기존 오즈 컬럼 제거 후 재조인: {existing_odds}")
        match_df = match_df.drop(columns=existing_odds)

    # 조인 키: MatchDate + HomeTeam + AwayTeam
    match_df["MatchDate"] = pd.to_datetime(match_df["MatchDate"], errors="coerce")
    odds_df["MatchDate"]  = pd.to_datetime(odds_df["MatchDate"],  errors="coerce")

    before_len = len(match_df)
    merged = match_df.merge(
        odds_df[["MatchDate", "HomeTeam", "AwayTeam"] + odds_cols],
        on=["MatchDate", "HomeTeam", "AwayTeam"],
        how="left",
    )

    matched = merged[odds_cols[0]].notna().sum()
    logger.info(f"  조인 결과: {len(merged)}행 (원본 {before_len}행), "
                f"오즈 매칭: {matched}건 ({matched/len(merged)*100:.1f}%)")

    # 오즈 없는 경우 결측 처리 (0.33 균등 확률로 채움 — 정보 없음 의미)
    for col in ["b365_home_prob", "b365_draw_prob", "b365_away_prob", "draw_market_prob"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.333)
    for col in ["market_home_fav", "odds_overround"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged.to_parquet(MATCH_FEAT_OUT, index=False)
    logger.info(f"  match_features.parquet 저장 완료: {merged.shape} → {MATCH_FEAT_OUT}")

    # 검증 출력
    logger.info("  피처 통계:")
    for col in odds_cols:
        if col in merged.columns:
            logger.info(f"    {col}: mean={merged[col].mean():.4f}, "
                        f"null={merged[col].isna().sum()}")


def main():
    logger.info("=" * 60)
    logger.info("P1 오즈 피처 추가 파이프라인 시작")
    logger.info("소스: Football-Data.co.uk (무료, CC0)")
    logger.info("=" * 60)

    odds_master = build_odds_master()
    if odds_master.empty:
        logger.error("오즈 데이터 수집 실패 — match_features 변경 없음")
        return

    # 오즈 마스터 캐시 저장
    odds_master_path = ROOT / "data" / "raw" / "odds_master.parquet"
    odds_master.to_parquet(odds_master_path, index=False)
    logger.info(f"오즈 마스터 저장: {odds_master_path}")

    merge_odds_into_features(odds_master)

    logger.info("=" * 60)
    logger.info("완료 — P1 모델 재학습 시 오즈 피처가 자동으로 사용됩니다.")
    logger.info("추가된 피처: b365_home_prob, b365_draw_prob, b365_away_prob,")
    logger.info("             draw_market_prob, market_home_fav, odds_overround")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
