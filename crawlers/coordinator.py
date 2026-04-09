"""Coordinator Agent - Orchestrates the crawling process.

Manages the overall crawling workflow:
1. Builds task queue (seasons -> teams -> players)
2. Tracks progress via checkpoint DB
3. Handles resume after interruption
4. Generates progress reports
"""

import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import SEASONS, CHECKPOINT_DB, FBREF_RAW_DIR
from crawlers.fbref_agent import FBrefAgent

logger = logging.getLogger(__name__)


class CrawlCoordinator:
    """Orchestrates the full crawling pipeline."""

    def __init__(self):
        self.fbref_agent = FBrefAgent()
        self.start_time = None

    def run(self, seasons=None, resume=True):
        """Run the full crawling pipeline.

        Args:
            seasons: List of seasons to crawl. Defaults to all.
            resume: If True, skip already completed tasks.
        """
        seasons = seasons or SEASONS
        self.start_time = datetime.now()

        print("=" * 60)
        print(f"EPL FBref Crawling - Starting at {self.start_time}")
        print(f"Seasons to crawl: {len(seasons)}")
        print(f"Resume mode: {resume}")
        print("=" * 60)

        if not resume:
            self._reset_checkpoints()

        for i, season in enumerate(seasons, 1):
            print(f"\n[{i}/{len(seasons)}] Crawling season: {season}")
            print("-" * 40)

            try:
                self.fbref_agent.crawl_season(season)
                self._print_progress()
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Progress has been saved.")
                print("Run again to resume from where you left off.")
                self._print_summary()
                self.fbref_agent.close()
                return
            except Exception as e:
                logger.error(f"Unexpected error in season {season}: {e}")
                print(f"ERROR in season {season}: {e}")
                print("Continuing to next season...")

        self._print_summary()
        self.fbref_agent.close()

    def _reset_checkpoints(self):
        """Clear all checkpoint data for fresh start."""
        with sqlite3.connect(CHECKPOINT_DB) as conn:
            conn.execute("DELETE FROM checkpoints WHERE source='fbref'")
        print("Checkpoints cleared.")

    def _print_progress(self):
        """Print current crawling progress."""
        progress = self.fbref_agent.get_progress()
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        total = completed + failed

        elapsed = datetime.now() - self.start_time
        print(f"  Progress: {completed} completed, {failed} failed ({total} total)")
        print(f"  Elapsed: {elapsed}")

    def _print_summary(self):
        """Print final crawling summary."""
        progress = self.fbref_agent.get_progress()
        elapsed = datetime.now() - self.start_time

        print("\n" + "=" * 60)
        print("CRAWLING SUMMARY")
        print("=" * 60)
        print(f"Completed: {progress.get('completed', 0)}")
        print(f"Failed:    {progress.get('failed', 0)}")
        print(f"Total time: {elapsed}")

        # Count saved files
        csv_count = len(list(FBREF_RAW_DIR.rglob("*.csv")))
        print(f"CSV files saved: {csv_count}")

        # List failed tasks
        if progress.get("failed", 0) > 0:
            print("\nFailed tasks:")
            with sqlite3.connect(CHECKPOINT_DB) as conn:
                rows = conn.execute(
                    "SELECT task_key, error_msg FROM checkpoints "
                    "WHERE source='fbref' AND status='failed'"
                ).fetchall()
                for key, err in rows:
                    print(f"  - {key}: {err}")

    def status(self):
        """Show current crawling status without running anything."""
        progress = self.fbref_agent.get_progress()
        print("Current FBref Crawling Status:")
        print(f"  Completed: {progress.get('completed', 0)}")
        print(f"  Failed:    {progress.get('failed', 0)}")

        csv_count = len(list(FBREF_RAW_DIR.rglob("*.csv")))
        print(f"  CSV files: {csv_count}")

        # Show which seasons have data
        if FBREF_RAW_DIR.exists():
            season_dirs = sorted(
                d.name for d in FBREF_RAW_DIR.iterdir() if d.is_dir()
            )
            print(f"  Seasons with data: {', '.join(season_dirs) or 'none'}")


# ==================================================================
# CLI Entry Point
# ==================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="EPL Crawl Coordinator")
    parser.add_argument(
        "--season", type=str, help="Crawl a specific season (e.g., '2000/01')"
    )
    parser.add_argument(
        "--from-season", type=str, help="Start from this season"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Start fresh (clear checkpoints)"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show status only"
    )
    args = parser.parse_args()

    coordinator = CrawlCoordinator()

    if args.status:
        coordinator.status()
    elif args.season:
        coordinator.run(seasons=[args.season], resume=not args.fresh)
    elif args.from_season:
        start_idx = SEASONS.index(args.from_season)
        coordinator.run(seasons=SEASONS[start_idx:], resume=not args.fresh)
    else:
        coordinator.run(resume=not args.fresh)
