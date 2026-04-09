"""Base crawler agent with Selenium browser, rate limiting, and checkpoint support.

Uses undetected-chromedriver to bypass Cloudflare bot protection.
"""

import time
import sqlite3
import logging
from pathlib import Path

import undetected_chromedriver as uc
from bs4 import BeautifulSoup

from config.settings import (
    FBREF_MIN_INTERVAL,
    CHECKPOINT_DB,
    LOG_DIR,
)
from crawlers.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class BaseCrawlerAgent:
    """Base class for all crawler agents.

    Provides:
    - Selenium browser with Cloudflare bypass (undetected-chromedriver)
    - Rate limiting
    - Checkpoint/resume via SQLite
    - Structured logging
    """

    def __init__(self, source_name: str, min_interval: float = 6.0):
        self.source_name = source_name
        self.rate_limiter = RateLimiter(min_interval=min_interval)
        self.driver = None  # lazy init
        self._init_checkpoint_db()
        self._setup_logging()

    def _get_driver(self):
        """Lazy-initialize the browser (so it's only created when needed)."""
        if self.driver is None:
            logger.info("Starting Chrome browser...")
            options = uc.ChromeOptions()
            # 최소 옵션만 사용 - 추가 옵션이 Cloudflare 감지를 유발함
            options.add_argument("--no-sandbox")
            # 설치된 Chrome 메이저 버전 자동 감지 → 매칭 ChromeDriver 사용
            import subprocess, re as _re
            _chrome_major = None
            try:
                _r = subprocess.run(
                    ["reg", "query",
                     r"HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon",
                     "/v", "version"],
                    capture_output=True, text=True,
                )
                _m = _re.search(r"(\d+)\.\d+\.\d+\.\d+", _r.stdout)
                _chrome_major = int(_m.group(1)) if _m else None
            except Exception:
                pass
            if _chrome_major:
                logger.info(f"Chrome 버전 감지: {_chrome_major}")
            self.driver = uc.Chrome(options=options, version_main=_chrome_major)
            self.driver.set_page_load_timeout(60)
            logger.info("Chrome browser started successfully")
        return self.driver

    def close(self):
        """Close the browser."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            logger.info("Browser closed")

    def _init_checkpoint_db(self):
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(CHECKPOINT_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    source TEXT,
                    task_key TEXT,
                    status TEXT DEFAULT 'pending',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    PRIMARY KEY (source, task_key)
                )
            """)

    def _setup_logging(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(
            LOG_DIR / f"{self.source_name}.log", encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Browser Fetch
    # ------------------------------------------------------------------
    def fetch(self, url: str, max_cf_wait: int = 30, max_retries: int = 3):
        """Fetch a page using Selenium, waiting for Cloudflare to pass.

        Args:
            url: The URL to fetch
            max_cf_wait: Max seconds to wait for Cloudflare challenge per attempt
            max_retries: Number of retry attempts if Cloudflare blocks

        Returns:
            BeautifulSoup object or None on failure
        """
        for attempt in range(max_retries):
            result = self._try_fetch(url, max_cf_wait)
            if result is not None:
                return result

            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                logger.warning(
                    f"Cloudflare blocked (attempt {attempt + 1}/{max_retries}). "
                    f"Waiting {wait_time}s before retry..."
                )
                # 브라우저를 닫았다가 다시 열어서 새 세션으로 시도
                self.close()
                time.sleep(wait_time)

        logger.error(f"All {max_retries} attempts failed for: {url}")
        return None

    def _try_fetch(self, url: str, max_cf_wait: int = 30):
        """Single fetch attempt."""
        self.rate_limiter.wait()
        driver = self._get_driver()

        try:
            logger.info(f"Fetching: {url}")
            driver.get(url)

            # 초기 로딩 대기
            time.sleep(3)

            # Cloudflare 챌린지 통과 대기
            passed = False
            for i in range(max_cf_wait):
                title = driver.title or ""

                # Cloudflare 챌린지 페이지 패턴들
                cf_patterns = ["잠시만", "moment", "checking", "attention", "verify"]
                is_cf = any(p in title.lower() for p in cf_patterns)

                if is_cf:
                    time.sleep(1)
                    continue
                elif title.strip() == "":
                    time.sleep(1)
                    continue
                else:
                    passed = True
                    break

            if not passed:
                logger.warning(f"Cloudflare challenge not passed ({max_cf_wait}s): {url}")
                return None

            # 페이지가 충분히 로드될 때까지 잠시 대기
            time.sleep(2)

            page_source = driver.page_source
            if len(page_source) < 1000:
                logger.warning(f"Page too short ({len(page_source)} chars): {url}")
                return None

            logger.info(f"OK: {url} ({len(page_source)} chars)")
            return BeautifulSoup(page_source, "lxml")

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            # 브라우저가 죽었을 수 있으니 재시작
            self.close()
            return None

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def is_completed(self, task_key: str) -> bool:
        with sqlite3.connect(CHECKPOINT_DB) as conn:
            row = conn.execute(
                "SELECT status FROM checkpoints WHERE source=? AND task_key=?",
                (self.source_name, task_key),
            ).fetchone()
            return row is not None and row[0] == "completed"

    def mark_completed(self, task_key: str):
        with sqlite3.connect(CHECKPOINT_DB) as conn:
            conn.execute(
                """INSERT INTO checkpoints (source, task_key, status)
                   VALUES (?, ?, 'completed')
                   ON CONFLICT(source, task_key) DO UPDATE SET
                       status='completed', updated_at=CURRENT_TIMESTAMP""",
                (self.source_name, task_key),
            )

    def mark_failed(self, task_key: str, error_msg: str = ""):
        with sqlite3.connect(CHECKPOINT_DB) as conn:
            conn.execute(
                """INSERT INTO checkpoints (source, task_key, status, error_msg)
                   VALUES (?, ?, 'failed', ?)
                   ON CONFLICT(source, task_key) DO UPDATE SET
                       status='failed', error_msg=?, updated_at=CURRENT_TIMESTAMP""",
                (self.source_name, task_key, error_msg, error_msg),
            )

    def get_progress(self) -> dict:
        with sqlite3.connect(CHECKPOINT_DB) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM checkpoints WHERE source=? GROUP BY status",
                (self.source_name,),
            ).fetchall()
            return dict(rows)
