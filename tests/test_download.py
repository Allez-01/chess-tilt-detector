"""Offline tests for the download step (no network calls)."""

import json
from datetime import datetime, timezone

from tilt.download import (
    archive_month,
    collect_recent_bullet_games,
    is_bullet_game,
    is_finished_month,
    load_month,
)

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)
URL = "https://api.chess.com/pub/player/mr_allez/games/{}/{:02d}"


class FakeClient:
    """Stands in for ChessComClient and counts archive downloads."""

    def __init__(self, archives: dict[str, list[dict]]):
        self.archives = archives
        self.calls: list[str] = []

    def archive_urls(self, username):
        return list(self.archives)

    def archive_games(self, url):
        self.calls.append(url)
        return self.archives[url]


def game(i, time_class="bullet", rules="chess"):
    return {"uuid": f"g{i}", "url": f"u{i}", "end_time": i, "time_class": time_class,
            "rules": rules, "pgn": "[%clk 0:00:59.9]"}


def test_archive_month():
    assert archive_month(URL.format(2025, 3)) == (2025, 3)


def test_finished_month():
    assert is_finished_month(2026, 9, NOW)
    assert not is_finished_month(2026, 10, NOW)


def test_bullet_filter():
    assert is_bullet_game(game(1))
    assert not is_bullet_game(game(1, time_class="blitz"))
    assert not is_bullet_game(game(1, rules="chess960"))


def test_finished_month_never_redownloaded(tmp_path):
    url = URL.format(2026, 9)
    client = FakeClient({url: [game(1)]})
    load_month(client, url, tmp_path, now=NOW)
    games, cached = load_month(client, url, tmp_path, now=NOW)
    assert cached and games == [game(1)]
    assert client.calls == [url]


def test_current_month_refreshed(tmp_path):
    url = URL.format(2026, 10)
    client = FakeClient({url: [game(1)]})
    load_month(client, url, tmp_path, now=NOW)
    assert json.loads((tmp_path / "2026-10.json").read_text())["complete"] is False
    load_month(client, url, tmp_path, now=NOW)
    assert client.calls == [url, url]


def test_collect_stops_early_and_keeps_newest(tmp_path):
    old, mid, new = URL.format(2026, 7), URL.format(2026, 8), URL.format(2026, 9)
    client = FakeClient({
        old: [game(1), game(2)],
        mid: [game(3), game(4, time_class="blitz"), game(5)],
        new: [game(6), game(7)],
    })
    games, stats = collect_recent_bullet_games(client, "Mr_Allez", 3, tmp_path)
    assert [g["uuid"] for g in games] == ["g5", "g6", "g7"]  # oldest first
    assert old not in client.calls  # enough games found before reaching it
