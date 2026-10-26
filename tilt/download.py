"""Step 1: download my most recent bullet games from the Chess.com Published-Data API.

Usage (from the project root):
    python -m tilt.download --username Mr_Allez --n-games 1000 --contact you@example.com

Writes:
    data/raw/archives/YYYY-MM.json     one cached file per monthly archive
    data/interim/bullet_games.jsonl    the newest N standard-chess bullet games, oldest first

Finished months are cached and never downloaded again. The current month is
always refreshed, because new games are still being added to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tilt.api import ChessComClient
from tilt.config import (
    ARCHIVE_CACHE_DIR,
    BULLET_GAMES_PATH,
    DEFAULT_N_GAMES,
    DEFAULT_USERNAME,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- cache helpers

def archive_month(archive_url: str) -> tuple[int, int]:
    """'.../games/2025/03' -> (2025, 3)."""
    year, month = archive_url.rstrip("/").split("/")[-2:]
    return int(year), int(month)


def is_finished_month(year: int, month: int, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    return (year, month) < (now.year, now.month)


def cache_path(year: int, month: int, cache_dir: Path) -> Path:
    return cache_dir / f"{year:04d}-{month:02d}.json"


def load_month(
    client: ChessComClient,
    archive_url: str,
    cache_dir: Path,
    now: datetime | None = None,
) -> tuple[list[dict], bool]:
    """Return (games, came_from_cache) for one monthly archive.

    A cache file is trusted only if it was saved after the month ended
    (complete=True). A month cached while still in progress is fetched once
    more after it finishes, then never again.
    """
    year, month = archive_month(archive_url)
    path = cache_path(year, month, cache_dir)

    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("complete"):
            return cached["games"], True

    games = client.archive_games(archive_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "archive_url": archive_url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "complete": is_finished_month(year, month, now),
                "games": games,
            }
        ),
        encoding="utf-8",
    )
    tmp.replace(path)  # atomic, so an interrupted run never leaves a half-written cache
    return games, False


# ---------------------------------------------------------------- filtering

def is_bullet_game(game: dict) -> bool:
    """Standard-chess bullet games only (drops variants such as Chess960 bullet)."""
    return game.get("time_class") == "bullet" and game.get("rules", "chess") == "chess"


def collect_recent_bullet_games(
    client: ChessComClient,
    username: str,
    n_games: int,
    cache_dir: Path,
) -> tuple[list[dict], dict]:
    """Walk monthly archives from newest to oldest until n_games bullet games are found."""
    urls = client.archive_urls(username)
    if not urls:
        raise SystemExit(f"No game archives found for '{username}'.")

    games_by_id: dict[str, dict] = {}
    stats = Counter()
    for url in sorted(urls, key=archive_month, reverse=True):
        month_games, cached = load_month(client, url, cache_dir)
        stats["months_cached" if cached else "months_downloaded"] += 1
        bullet = [g for g in month_games if is_bullet_game(g)]
        for g in bullet:
            games_by_id[g.get("uuid") or g["url"]] = g
        log.info(
            "%s: %d games, %d bullet (%s)",
            "-".join(f"{p:02d}" for p in archive_month(url)),
            len(month_games), len(bullet), "cache" if cached else "downloaded",
        )
        if len(games_by_id) >= n_games:
            break

    games = sorted(games_by_id.values(), key=lambda g: g.get("end_time", 0))
    return games[-n_games:], dict(stats)


# ---------------------------------------------------------------- output

def write_jsonl(games: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")


def my_side(game: dict, username: str) -> str | None:
    for side in ("white", "black"):
        if game.get(side, {}).get("username", "").lower() == username.lower():
            return side
    return None


def summarise(games: list[dict], username: str, stats: dict) -> str:
    if not games:
        return "No bullet games found."
    ends = [datetime.fromtimestamp(g["end_time"], timezone.utc) for g in games]
    colours = Counter(my_side(g, username) for g in games)
    tcs = Counter(g.get("time_control") for g in games)
    with_clk = sum("%clk" in g.get("pgn", "") for g in games)
    lines = [
        "=== Download summary ===",
        f"Months downloaded: {stats.get('months_downloaded', 0)}, "
        f"from cache: {stats.get('months_cached', 0)}",
        f"Bullet games kept: {len(games)}",
        f"Date range (UTC):  {min(ends):%Y-%m-%d} to {max(ends):%Y-%m-%d}",
        "Time controls:     " + ", ".join(f"{k}: {v}" for k, v in tcs.most_common()),
        "My colour:         " + ", ".join(f"{k}: {v}" for k, v in colours.most_common()),
        f"Games with %clk:   {with_clk} / {len(games)}",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--username", default=DEFAULT_USERNAME)
    p.add_argument("--n-games", type=int, default=DEFAULT_N_GAMES)
    p.add_argument(
        "--contact",
        default=os.environ.get("CHESSCOM_CONTACT"),
        help="Email or handle for the User-Agent header (or set CHESSCOM_CONTACT).",
    )
    p.add_argument("--cache-dir", type=Path, default=ARCHIVE_CACHE_DIR)
    p.add_argument("--out", type=Path, default=BULLET_GAMES_PATH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    contact = args.contact or f"chess.com user {args.username}"

    client = ChessComClient(contact=contact)
    games, stats = collect_recent_bullet_games(
        client, args.username, args.n_games, args.cache_dir
    )
    write_jsonl(games, args.out)
    print(summarise(games, args.username, stats))
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
