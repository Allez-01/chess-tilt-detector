"""Tests for the weekly report and shared insights, on a small synthetic history."""

import numpy as np
import pandas as pd

from tilt.insights import standing_findings, weekly_comparison
from tilt.weekly_report import build_report


def history(n_days=28, per_day=8, seed=42):
    """Older weeks: fast openings, few time losses. Last week: slow openings, many time losses."""
    rng = np.random.default_rng(seed)
    rows, moves = [], []
    for d in range(n_days):
        last_week = d >= n_days - 7
        for i in range(per_day):
            gid = d * per_day + i
            lost_on_time = rng.random() < (0.6 if last_week else 0.2)
            rows.append({
                "game_id": gid, "start": pd.Timestamp("2026-09-01", tz="UTC") + pd.Timedelta(days=d, minutes=3 * i),
                "won": int(not lost_on_time and rng.random() < 0.6), "lost_on_time": int(lost_on_time),
                "opening_time_used": rng.normal(18 if last_week else 9, 2),
                "game_index_in_session": i + 1, "loss_streak": int(rng.integers(0, 3)),
            })
            moves.append(pd.DataFrame({"game_id": gid, "clock_before": rng.uniform(0, 60, 30),
                                       "is_blunder_wc": (rng.random(30) < 0.1).astype(int)}))
    return pd.DataFrame(rows), pd.concat(moves)


def test_week_window_and_baseline():
    games, moves = history()
    week, metrics, info = weekly_comparison(games, moves)
    assert info["week_games"] == 7 * 8  # (end - 7 days, end]: the last 7 days of games
    assert info["baseline_games"] + info["week_games"] == len(games)
    by_name = {m.name: m for m in metrics}
    assert by_name["Games lost on time"].worse_by > 0.2


def test_report_flags_planted_weaknesses():
    games, moves = history()
    report = build_report(games, moves)
    weak = report.split("## Your weaknesses this week")[1].split("##")[0]
    assert "Games lost on time" in weak and "Seconds used on first 10 moves" in weak
    assert "Post-game analysis of finished games only." in report


def test_standing_findings_are_computed():
    games, _ = history()
    lines = standing_findings(games)
    assert len(lines) == 3 and all("%" in line for line in lines)
