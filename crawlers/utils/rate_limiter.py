"""Rate limiter for polite crawling."""

import time
import threading


class RateLimiter:
    """Thread-safe rate limiter with minimum interval between requests."""

    def __init__(self, min_interval: float = 6.0):
        self.min_interval = min_interval
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Block until it's safe to make the next request."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def update_interval(self, new_interval: float):
        """Dynamically adjust the rate limit (e.g., after 429 response)."""
        with self._lock:
            self.min_interval = new_interval
