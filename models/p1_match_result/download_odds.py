"""
Football-Data.co.uk EPL 베팅 odds CSV 수집 스크립트
==================================================
- URL: https://www.football-data.co.uk/mmz4281/{시즌}/E0.csv
- 시즌 형식: 0001 (2000/01) ~ 2425 (2024/25)
- 저장 경로: data/raw/betting_odds/E0_{시즌}.csv
- 레이트 리밋: 2초
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "betting_odds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
RATE_LIMIT = 2.0  # 초

# 2000/01 ~ 2024/25
SEASONS: list[tuple[str, str]] = []
for start in range(0, 25):  # 0..24
    yy_start = f"{start:02d}"
    yy_end = f"{(start + 1) % 100:02d}"
    season_code = f"{yy_start}{yy_end}"
    season_label = f"20{yy_start}/{yy_end}"
    SEASONS.append((season_code, season_label))


def download_one(season_code: str, season_label: str) -> bool:
    out_file = OUT_DIR / f"E0_{season_code}.csv"
    if out_file.exists() and out_file.stat().st_size > 1024:
        print(f"  [skip] {season_label} → 이미 존재")
        return True

    url = BASE_URL.format(season_code=season_code)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        out_file.write_bytes(resp.content)
        print(f"  [ok]   {season_label} → {out_file.name} ({len(resp.content) / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [fail] {season_label}: {e}")
        return False


def main():
    print("=" * 70)
    print("FOOTBALL-DATA.CO.UK ODDS DOWNLOADER")
    print("=" * 70)
    print(f"  대상 시즌: {len(SEASONS)}개")
    print(f"  저장 경로: {OUT_DIR}")

    ok = 0
    fail = 0
    for season_code, season_label in SEASONS:
        success = download_one(season_code, season_label)
        if success:
            ok += 1
        else:
            fail += 1
        time.sleep(RATE_LIMIT)

    print("\n" + "=" * 70)
    print(f"완료: {ok}/{len(SEASONS)} 성공, {fail} 실패")
    print("=" * 70)


if __name__ == "__main__":
    main()
