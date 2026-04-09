"""
Transfermarkt 크롤링 실행 스크립트
====================================
VS Code에서 바로 실행 (FBref 크롤링과 별도 터미널에서 동시 실행 가능)

사용법:
    python run_crawl_tm.py                     # 전체 시즌 (2000/01 ~ 2024/25)
    python run_crawl_tm.py --season 2023/24    # 특정 시즌만
    python run_crawl_tm.py --from-season 2020/21  # 특정 시즌부터
    python run_crawl_tm.py --status            # 진행 상황 확인
    python run_crawl_tm.py --fresh             # 처음부터 다시 시작
"""

import sys
import os
import argparse
import logging
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SEASONS, CHECKPOINT_DB, TM_RAW_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Transfermarkt EPL 크롤링",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_crawl_tm.py                       전체 시즌 크롤링
  python run_crawl_tm.py --season 2023/24      특정 시즌만
  python run_crawl_tm.py --from-season 2020/21 2020/21부터
  python run_crawl_tm.py --status              진행 상황 확인
  python run_crawl_tm.py --fresh               처음부터 다시
        """,
    )
    parser.add_argument("--season", type=str, help="특정 시즌 (예: 2023/24)")
    parser.add_argument("--from-season", type=str, help="이 시즌부터 (예: 2020/21)")
    parser.add_argument("--fresh", action="store_true", help="체크포인트 초기화")
    parser.add_argument("--status", action="store_true", help="진행 상황만 확인")
    args = parser.parse_args()

    # logs 디렉토리
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(log_dir, "crawl_tm.log"), encoding="utf-8"
            ),
        ],
    )

    if args.status:
        _show_status()
        return

    from crawlers.transfermarkt_agent import TransfermarktAgent

    agent = TransfermarktAgent()

    if args.fresh:
        with sqlite3.connect(CHECKPOINT_DB) as conn:
            conn.execute("DELETE FROM checkpoints WHERE source='transfermarkt'")
        print("Transfermarkt checkpoints cleared.")

    try:
        if args.season:
            print(f"\n>>> TM: 시즌 {args.season} 크롤링 시작")
            agent.crawl_season(args.season)
        elif args.from_season:
            start_idx = SEASONS.index(args.from_season)
            remaining = SEASONS[start_idx:]
            print(f"\n>>> TM: {args.from_season}부터 {len(remaining)}시즌 크롤링")
            agent.crawl_all_seasons(remaining)
        else:
            print(f"\n>>> TM: 전체 {len(SEASONS)}시즌 크롤링 시작")
            agent.crawl_all_seasons()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved. Run again to resume.")
    finally:
        agent.close()

    print("\n>>> Transfermarkt 크롤링 완료!")


def _show_status():
    """Show current Transfermarkt crawling status."""
    with sqlite3.connect(CHECKPOINT_DB) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM checkpoints "
            "WHERE source='transfermarkt' GROUP BY status"
        ).fetchall()
        progress = dict(rows) if rows else {}

    print("Transfermarkt Crawling Status:")
    print(f"  Completed: {progress.get('completed', 0)}")
    print(f"  Failed:    {progress.get('failed', 0)}")

    if TM_RAW_DIR.exists():
        csv_count = len(list(TM_RAW_DIR.rglob("*.csv")))
        season_dirs = sorted(d.name for d in TM_RAW_DIR.iterdir() if d.is_dir())
        print(f"  CSV files: {csv_count}")
        print(f"  Seasons:   {', '.join(season_dirs) or 'none'}")
    else:
        print("  No data yet")


if __name__ == "__main__":
    main()
