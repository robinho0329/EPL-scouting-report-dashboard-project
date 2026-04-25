"""scripts/p8_merge_crossleague.py — 해외→EPL 크로스리그 이적 데이터를 transfer_dataset에 병합.

실행 순서:
  1. python scripts/p8_crossleague_crawler.py    (크롤링, ~10-15분)
  2. python scripts/p8_merge_crossleague.py      (병합, 수초)
  3. python models/p8_transfer_adapt/train.py    (재학습)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
EXT_PATH = ROOT / "data" / "raw" / "p8_crossleague_transfers.parquet"
ORG_PATH = ROOT / "models" / "p8_transfer_adaptation_archived" / "transfer_dataset.parquet"
OUT_PATH = ROOT / "models" / "p8_transfer_adaptation_archived" / "transfer_dataset.parquet"
BKP_PATH = ROOT / "models" / "p8_transfer_adaptation_archived" / "transfer_dataset_backup.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("p8_merge_cross")


def main() -> None:
    log.info("=" * 60)
    log.info("🔀 P8 크로스리그 이적 데이터 병합")
    log.info("=" * 60)

    # 외부 데이터 확인
    if not EXT_PATH.exists():
        log.error(f"❌ 크로스리그 데이터 없음: {EXT_PATH}")
        log.error("   먼저 python scripts/p8_crossleague_crawler.py 를 실행하세요.")
        sys.exit(1)

    df_ext = pd.read_parquet(EXT_PATH)
    log.info(f"📂 크로스리그 데이터: {len(df_ext)}건")
    log.info(f"   리그별: {df_ext['source_league'].value_counts().to_dict()}")
    log.info(f"   team_old==team_new 비율: "
             f"{(df_ext['team_old']==df_ext['team_new']).mean()*100:.1f}% (0이어야 정상)")

    df_org = pd.read_parquet(ORG_PATH)
    log.info(f"📂 기존 데이터: {len(df_org)}건 (EPL 내부)")

    # 백업
    df_org.to_parquet(BKP_PATH, index=False, engine="pyarrow")
    log.info(f"💾 백업 저장: transfer_dataset_backup.parquet")

    # 기존 데이터에 source_league / league_level_diff 컬럼이 없으면 추가
    if "source_league" not in df_org.columns:
        df_org["source_league"] = "EPL"
        log.info("  기존 데이터에 source_league='EPL' 컬럼 추가")
    if "league_level_diff" not in df_org.columns:
        df_org["league_level_diff"] = 0
        log.info("  기존 데이터에 league_level_diff=0 컬럼 추가")

    # 컬럼 맞추기: 기존에 없는 컬럼은 NaN으로 추가
    new_cols = [c for c in df_ext.columns if c not in df_org.columns]
    if new_cols:
        log.info(f"  새 컬럼 추가 ({len(new_cols)}개): {new_cols}")
        for col in new_cols:
            df_org[col] = None

    # 외부 데이터를 기존 컬럼 순서에 맞게 정렬
    df_ext_aligned = df_ext.reindex(columns=df_org.columns)

    # 병합
    df_merged = pd.concat([df_org, df_ext_aligned], ignore_index=True)

    # 중복 제거
    before = len(df_merged)
    df_merged = df_merged.drop_duplicates(
        subset=["player", "season_old", "season_new", "team_old"]
    )
    log.info(f"  중복 제거: {before} → {len(df_merged)}건")

    # 저장
    df_merged.to_parquet(OUT_PATH, index=False, engine="pyarrow")

    log.info(f"\n✅ 병합 완료!")
    log.info(f"   기존: {len(df_org)}건 → 병합 후: {len(df_merged)}건")
    log.info(f"   추가: +{len(df_merged) - len(df_org)}건")
    log.info(f"\n   source_league 분포:")
    log.info(f"   {df_merged['source_league'].value_counts().to_string()}")
    log.info(f"\n   adapted=1 비율: {df_merged['adapted'].mean()*100:.1f}%")
    log.info(f"\n   team_old==team_new 건수 확인 (0이어야 크로스리그 정상):")
    same_team = (df_merged['team_old'] == df_merged['team_new']).sum()
    log.info(f"   {same_team}건 ({same_team/len(df_merged)*100:.1f}%)")

    log.info("\n다음 단계: python models/p8_transfer_adapt/train.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
