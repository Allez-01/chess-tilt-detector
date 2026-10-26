"""Step 5b: weekly "bullet weaknesses" report (Markdown), this week vs my earlier games.

Usage (from the project root, after steps 1-4):
    python -m tilt.weekly_report                     # week ending at my latest game
    python -m tilt.weekly_report --week-end 2026-10-25

Writes reports/weekly/bullet_weaknesses_<week-end>.md and prints it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from tilt.config import REPORTS_DIR
from tilt.insights import (
    MIN_WEEK_GAMES, Metric, blunder_rate_by_clock, load_games, load_my_moves, standing_findings,
    weekly_comparison,
)

WORSE_THRESHOLD = 0.10  # flag a metric when it is at least 10% worse than baseline (relative)


def relative_worse(m: Metric) -> float:
    if any(map(math.isnan, (m.week, m.baseline))) or m.baseline == 0:
        return float("nan")
    return m.worse_by / abs(m.baseline)


def build_report(games: pd.DataFrame, my_moves: pd.DataFrame,
                 week_end: pd.Timestamp | None = None) -> str:
    week, metrics, info = weekly_comparison(games, my_moves, week_end)
    lines = [
        f"# Bullet weaknesses: week ending {info['week_end']:%Y-%m-%d}",
        "",
        f"{info['week_games']} games this week, compared with {info['baseline_games']} earlier games.",
        "Post-game analysis of finished games only.",
        "",
    ]
    if info["week_games"] < MIN_WEEK_GAMES:
        lines += [f"> Only {info['week_games']} games this week, so treat the comparison as a rough "
                  "signal, not a finding.", ""]

    lines += ["## This week vs your usual", "", "| Measure | This week | Usual | |", "|---|---|---|---|"]
    for m in metrics:
        r = relative_worse(m)
        flag = "" if math.isnan(r) else ("worse" if r >= WORSE_THRESHOLD
                                         else "better" if r <= -WORSE_THRESHOLD else "same")
        fmt = lambda v: "-" if math.isnan(v) else m.fmt.format(v)
        lines.append(f"| {m.name} | {fmt(m.week)} | {fmt(m.baseline)} | {flag} |")

    flagged = sorted((m for m in metrics if relative_worse(m) >= WORSE_THRESHOLD),
                     key=relative_worse, reverse=True)[:3]
    lines += ["", "## Your weaknesses this week", ""]
    if flagged:
        lines += [f"{i}. **{m.name}** ({m.fmt.format(m.week)} vs {m.fmt.format(m.baseline)} usually). "
                  f"{m.advice}" for i, m in enumerate(flagged, 1)]
    else:
        lines.append("Nothing is more than 10% worse than usual this week.")

    clock = blunder_rate_by_clock(my_moves[my_moves["game_id"].isin(set(week["game_id"]))])
    lines += ["", "## Blunder rate by clock this week", "", "| Clock left | Blunder rate | Moves |",
              "|---|---|---|"]
    lines += [f"| {r.clock} | {'-' if r.moves == 0 else f'{r.blunder_rate:.1%}'} | {r.moves} |"
              for r in clock.itertuples()]
    lines += ["", "## Patterns from your full history", "",
              *(f"- {a}" for a in standing_findings(games))]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--week-end", default=None, help="YYYY-MM-DD (UTC); default: latest game")
    p.add_argument("--out-dir", type=Path, default=REPORTS_DIR / "weekly")
    args = p.parse_args(argv)

    games, my_moves = load_games(), load_my_moves()
    week_end = (pd.Timestamp(args.week_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                if args.week_end else None)
    report = build_report(games, my_moves, week_end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (week_end or games["start"].max()).strftime("%Y-%m-%d")
    path = args.out_dir / f"bullet_weaknesses_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
