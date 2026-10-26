"""Polite client for the Chess.com Published-Data API.

Rules this client follows (see https://www.chess.com/news/view/published-data-api):
- one request at a time, never in parallel, with a short pause between calls
- a User-Agent that names the script and gives a way to contact its owner
- on 429 "too many requests" (or a transient 5xx), pause and retry with backoff
"""

from __future__ import annotations

import logging
import time

import requests

from tilt import __version__

log = logging.getLogger(__name__)

BASE_URL = "https://api.chess.com/pub"


class ChessComError(RuntimeError):
    """Raised when the API returns an error we can't recover from."""


class ChessComClient:
    def __init__(
        self,
        contact: str,
        min_interval: float = 1.0,
        max_retries: int = 5,
        backoff_seconds: float = 30.0,
        timeout: float = 30.0,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    f"chess-tilt-detector/{__version__} "
                    f"(personal post-game analysis script; contact: {contact})"
                ),
                "Accept": "application/json",
            }
        )
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout
        self._last_request = 0.0

    def _wait_turn(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get_json(self, url: str) -> dict:
        """GET a URL and return its JSON, retrying on 429 and 5xx."""
        for attempt in range(1, self.max_retries + 1):
            self._wait_turn()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise ChessComError(f"Network error on {url}: {exc}") from exc
                wait = 5.0 * attempt
                log.warning("Network error on %s (%s); retrying in %.0fs", url, exc, wait)
                time.sleep(wait)
                continue
            finally:
                self._last_request = time.monotonic()

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else self.backoff_seconds * attempt
                )
                log.warning(
                    "HTTP %s on %s (attempt %d/%d); pausing %.0fs",
                    resp.status_code, url, attempt, self.max_retries, wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                raise ChessComError(f"Not found (404): {url} - check the username")
            raise ChessComError(f"HTTP {resp.status_code} on {url}: {resp.text[:200]}")

        raise ChessComError(f"Gave up on {url} after {self.max_retries} attempts")

    def archive_urls(self, username: str) -> list[str]:
        """Monthly archive URLs for a player, oldest first."""
        data = self.get_json(f"{BASE_URL}/player/{username.lower()}/games/archives")
        return data.get("archives", [])

    def archive_games(self, archive_url: str) -> list[dict]:
        """All games in one monthly archive."""
        return self.get_json(archive_url).get("games", [])
