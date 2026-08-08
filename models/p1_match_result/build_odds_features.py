"""
Football-Data odds CSV → 통합 odds 피처 Parquet 변환
=====================================================
- B365H/D/A 우선, 없으면 WH/LB/GB 등 fallback
- implied probability 정규화 (오버라운드 제거)
- 팀명 정규화 (Football-Data ↔ EPL 프로젝트)
- 키: Season + MatchDate + HomeTeam + AwayTeam
- 출력: data/processed/match_odds.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ODDS_DIR = PROJECT_ROOT / "data" / "raw" / "betting_odds"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "match_results.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "match_odds.parquet"

# Football-Data 팀명 → EPL 프로젝트 표준 (이미 거의 일치하지만 안전망)
TEAM_NAME_MAP = {
    # 동일하지만 명시
    "Man United": "Man United",
    "Man City": "Man City",
    "Newcastle": "Newcastle",
    "Tottenham": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
    "Nott'm Forest": "Nott'm Forest",
    # 가능한 변형
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Norwich City": "Norwich",
    "Cardiff City": "Cardiff",
    "Swansea City": "Swansea",
    "Hull City": "Hull",
    "Stoke City": "Stoke",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "West Bromwich Albion": "West Brom",
    "West Brom": "West Brom",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Weds",
    "Sheffield Wednesday": "Sheffield Weds",
    "Bolton Wanderers": "Bolton",
    "Bradford City": "Bradford",
    "Wigan Athletic": "Wigan",
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Charlton Athletic": "Charlton",
    "Coventry City": "Coventry",
    "Ipswich Town": "Ipswich",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "QPR": "QPR",
    "AFC Bournemouth": "Bournemouth",
}


def normalize_team(name) -> str | None:
    if pd.isna(name):
        return None
    s = str(name).strip()
    return TEAM_NAME_MAP.get(s, s)


def parse_date(s):
    """Football-Data 날짜는 dd/mm/yy 또는 dd/mm/yyyy."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


# odds 컬럼 우선순위
ODDS_TRIPLES = [
    ("B365H", "B365D", "B365A"),
    ("BWH", "BWD", "BWA"),
    ("WHH", "WHD", "WHA"),
    ("LBH", "LBD", "LBA"),
    ("IWH", "IWD", "IWA"),
    ("PSH", "PSD", "PSA"),
    ("VCH", "VCD", "VCA"),
    ("GBH", "GBD", "GBA"),
    ("SBH", "SBD", "SBA"),
    ("SYH", "SYD", "SYA"),
    ("SOH", "SOD", "SOA"),
]


def pick_odds(row: pd.Series) -> tuple[float, float, float, str]:
    """선호 베팅사부터 사용 가능한 odds 추출."""
    for h, d, a in ODDS_TRIPLES:
        if h in row.index and d in row.index and a in row.index:
            vh, vd, va = row[h], row[d], row[a]
            if pd.notna(vh) and pd.notna(vd) and pd.notna(va) and vh > 0 and vd > 0 and va > 0:
                return float(vh), float(vd), float(va), h.replace("H", "")
    return np.nan, np.nan, np.nan, ""


def load_one(path: Path) -> pd.DataFrame:
    """단일 시즌 CSV → Date/Home/Away/odds 컬럼만 추출."""
    df = pd.read_csv(path, encoding="latin1")
    needed_base = {"Date", "HomeTeam", "AwayTeam"}
    if not needed_base.issubset(df.columns):
        return pd.DataFrame()

    # 사용 가능한 odds 컬럼만 남김
    keep = ["Date", "HomeTeam", "AwayTeam"]
    for h, d, a in ODDS_TRIPLES:
        if h in df.columns:
            keep += [h, d, a]
    df = df[keep].copy()

    # 행마다 odds 추출
    odds_records = df.apply(pick_odds, axis=1, result_type="expand")
    df["odds_h"] = odds_records[0]
    df["odds_d"] = odds_records[1]
    df["odds_a"] = odds_records[2]
    df["odds_source"] = odds_records[3]
    df = df[["Date", "HomeTeam", "AwayTeam", "odds_h", "odds_d", "odds_a", "odds_source"]]
    df["MatchDate"] = df["Date"].apply(parse_date)
    df = df.drop(columns=["Date"])
    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team)
    return df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam"])


def add_implied_features(df: pd.DataFrame) -> pd.DataFrame:
    """odds → implied probability + 정규화 + 파생 피처."""
    df = df.copy()
    inv_h = 1.0 / df["odds_h"]
    inv_d = 1.0 / df["odds_d"]
    inv_a = 1.0 / df["odds_a"]
    overround = inv_h + inv_d + inv_a

    df["imp_h"] = inv_h
    df["imp_d"] = inv_d
    df["imp_a"] = inv_a
    df["overround"] = overround
    df["norm_h"] = inv_h / overround
    df["norm_d"] = inv_d / overround
    df["norm_a"] = inv_a / overround

    eps = 1e-6
    df["odds_ratio_h_a"] = df["norm_h"] / (df["norm_a"] + eps)
    df["odds_ratio_h_d"] = df["norm_h"] / (df["norm_d"] + eps)
    df["log_odds_h"] = np.log(df["odds_h"].clip(lower=1.01))
    df["log_odds_d"] = np.log(df["odds_d"].clip(lower=1.01))
    df["log_odds_a"] = np.log(df["odds_a"].clip(lower=1.01))
    df["log_odds_diff_h_a"] = df["log_odds_h"] - df["log_odds_a"]
    df["norm_h_minus_a"] = df["norm_h"] - df["norm_a"]
    # entropy(불확실성)
    p = df[["norm_h", "norm_d", "norm_a"]].values.clip(min=eps)
    df["odds_entropy"] = -(p * np.log(p)).sum(axis=1)
    return df


def main():
    print("=" * 70)
    print("BUILD ODDS FEATURES")
    print("=" * 70)

    csv_files = sorted(ODDS_DIR.glob("E0_*.csv"))
    print(f"  CSV 파일: {len(csv_files)}개")

    frames = []
    for f in csv_files:
        try:
            d = load_one(f)
            if not d.empty:
                d["__src__"] = f.name
                frames.append(d)
                print(f"  [ok] {f.name}: rows={len(d)}, odds_avail={d['odds_h'].notna().sum()}")
        except Exception as e:
            print(f"  [fail] {f.name}: {e}")

    odds_all = pd.concat(frames, ignore_index=True)
    print(f"\n  통합 행 수: {len(odds_all)}")
    print(f"  odds 채워진 행: {odds_all['odds_h'].notna().sum()} ({odds_all['odds_h'].notna().mean()*100:.1f}%)")

    odds_all = add_implied_features(odds_all)

    # 결과 매칭률 점검
    res = pd.read_parquet(RESULTS_PATH)
    res["MatchDate"] = pd.to_datetime(res["MatchDate"])
    merged = res.merge(
        odds_all[["MatchDate", "HomeTeam", "AwayTeam", "odds_h", "odds_d", "odds_a"]],
        on=["MatchDate", "HomeTeam", "AwayTeam"],
        how="left",
    )
    match_rate = merged["odds_h"].notna().mean()
    print(f"\n  match_results 매칭률: {match_rate * 100:.2f}% ({merged['odds_h'].notna().sum()}/{len(merged)})")

    if match_rate < 0.9:
        # 팀명 미매칭 진단
        unmatched = merged[merged["odds_h"].isna()]
        bad_teams = pd.concat([unmatched["HomeTeam"], unmatched["AwayTeam"]]).value_counts().head(15)
        print("\n  [경고] 매칭 실패 상위 팀:")
        print(bad_teams)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    odds_all.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"\n  저장: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
