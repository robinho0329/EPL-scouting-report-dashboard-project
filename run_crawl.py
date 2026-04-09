"""
EPL FBref 크롤링 실행 스크립트
================================
VS Code에서 바로 실행하세요 (F5 또는 터미널에서 python run_crawl.py)

사용법:
    python run_crawl.py                    # 전체 시즌 (2000/01 ~ 2024/25)
    python run_crawl.py --season 2023/24   # 특정 시즌만
    python run_crawl.py --from-season 2020/21  # 특정 시즌부터
    python run_crawl.py --status           # 현재 진행 상황 확인
    python run_crawl.py --fresh            # 처음부터 다시 시작
"""

import sys
import os
import argparse
import logging

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SEASONS
from crawlers.coordinator import CrawlCoordinator


def main():
    parser = argparse.ArgumentParser(
        description="EPL FBref 크롤링",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_crawl.py                      전체 시즌 크롤링
  python run_crawl.py --season 2023/24     특정 시즌만
  python run_crawl.py --from-season 2020/21  2020/21부터
  python run_crawl.py --status             진행 상황 확인
  python run_crawl.py --fresh              처음부터 다시
        """,
    )
    parser.add_argument("--season", type=str, help="특정 시즌 크롤링 (예: 2023/24)")
    parser.add_argument("--from-season", type=str, help="이 시즌부터 크롤링 (예: 2020/21)")
    parser.add_argument("--fresh", action="store_true", help="체크포인트 초기화 후 처음부터")
    parser.add_argument("--status", action="store_true", help="진행 상황만 확인")
    args = parser.parse_args()

    # logs 디렉토리 생성 (로깅 설정 전에 먼저)
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(log_dir, "crawl.log"),
                encoding="utf-8",
            ),
        ],
    )

    coordinator = CrawlCoordinator()

    if args.status:
        coordinator.status()
        return

    if args.season:
        print(f"\n>>> 시즌 {args.season} 크롤링 시작")
        coordinator.run(seasons=[args.season], resume=not args.fresh)
    elif args.from_season:
        start_idx = SEASONS.index(args.from_season)
        remaining = SEASONS[start_idx:]
        print(f"\n>>> {args.from_season}부터 {len(remaining)}시즌 크롤링 시작")
        coordinator.run(seasons=remaining, resume=not args.fresh)
    else:
        print(f"\n>>> 전체 {len(SEASONS)}시즌 크롤링 시작")
        coordinator.run(resume=not args.fresh)

    # 크롤링 완료 후 자동으로 데이터 집계
    print("\n" + "=" * 60)
    print("크롤링 완료! 데이터 집계를 시작합니다...")
    print("=" * 60)

    from pipeline.aggregate import run_pipeline
    run_pipeline()

    print("\n>>> 모든 작업 완료!")
    print(">>> 대시보드 실행: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
