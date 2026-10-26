"""Shared analysis for the weekly report and the dashboard (finished games only)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tilt.config import LABELLED_MOVES_PATH, SESSION_GAMES_PATH

CLOCK_BINS = [0, 3, 5, 10, 30, 61]
CLOCK_LABELS = ["<3s", "3-5s", "5-10s", "10-30s", "30s+"]
SLOW_OPENING_SECONDS = 15
MIN_WEEK_GAMES = 10


def load_games(path=SESSION_GAMES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["utc_datetime"], utc=True, format="ISO8601")
    return df.sort_values("start").reset_index(drop=True)


def load_my_moves(path=LABELLED_MOVES_PATH) -> pd.DataFrame:
    moves = pd.read_csv(path)
    return moves[moves["is_me"].astype(bool)].copy()


def blunder_rate_by_clock(my_moves: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(my_moves["clock_before"], CLOCK_BINS, labels=CLOCK_LABELS, right=False)
    return (my_moves.groupby(b, observed=False)["is_blunder_wc"]
            .agg(blunder_rate="mean", moves="size").rename_axis("clock").reset_index())


def blunder_rate_by_phase(my_moves: pd.DataFrame) -> pd.DataFrame:
    return (my_moves.groupby("phase")["is_blunder_wc"].agg(blunder_rate="mean", moves="size")
            .reindex(["opening", "middlegame", "endgame"]).rename_axis("phase").reset_index())


def win_rate_by_game_number(games: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(games["game_index_in_session"], [1, 2, 6, 11, 21, float("inf")], right=False,
               labels=["1st", "2-5", "6-10", "11-20", "21+"])
    return (games.groupby(b, observed=False)
            .agg(win_rate=("won", "mean"), lost_on_time=("lost_on_time", "mean"), games=("won", "size"))
            .rename_axis("game in session").reset_index())


def win_rate_by_loss_streak(games: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(games["loss_streak"], [0, 1, 2, 3, float("inf")], right=False,
               labels=["0", "1", "2", "3+"])
    return (games.groupby(b, observed=False).agg(win_rate=("won", "mean"), games=("won", "size"))
            .rename_axis("losses in a row").reset_index())


# ---------------------------------------------------------------- weekly comparison

@dataclass
class Metric:
    name: str
    week: float
    baseline: float
    higher_is_better: bool
    advice: str
    fmt: str = "{:.0%}"

    @property
    def worse_by(self) -> float:
        """Positive when this week is worse than the baseline (in the metric's own units)."""
        d = self.week - self.baseline
        return -d if self.higher_is_better else d


def game_metrics(games: pd.DataFrame, my_moves: pd.DataFrame) -> dict[str, float]:
    ids = set(games["game_id"])
    mv = my_moves[my_moves["game_id"].isin(ids)]
    slow = games["opening_time_used"] >= SLOW_OPENING_SECONDS
    first = games["game_index_in_session"] == 1
    return {
        "win_rate": games["won"].mean(),
        "lost_on_time": games["lost_on_time"].mean(),
        "blunder_rate": mv["is_blunder_wc"].mean(),
        "low_clock_blunder_rate": mv.loc[mv["clock_before"] < 10, "is_blunder_wc"].mean(),
        "opening_time_used": games["opening_time_used"].mean(),
        "slow_opening_share": slow.mean(),
        "first_game_win_rate": games.loc[first, "won"].mean() if first.any() else float("nan"),
    }


def weekly_comparison(games: pd.DataFrame, my_moves: pd.DataFrame,
                      week_end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[Metric], dict]:
    """This week (7 days up to week_end, default: latest game) vs every earlier game."""
    week_end = week_end or games["start"].max()
    week_start = week_end - pd.Timedelta(days=7)
    week = games[(games["start"] > week_start) & (games["start"] <= week_end)]
    base = games[games["start"] <= week_start]
    if base.empty:  # not enough history: compare against all games
        base = games
    w, b = game_metrics(week, my_moves), game_metrics(base, my_moves)
    metrics = [
        Metric("Win rate", w["win_rate"], b["win_rate"], True,
               "Results are down; check the time-loss and opening lines below for the cause."),
        Metric("Games lost on time", w["lost_on_time"], b["lost_on_time"], False,
               "Flag less: premove recaptures and simplify when you're ahead on the board."),
        Metric("Blunder rate (all my moves)", w["blunder_rate"], b["blunder_rate"], False,
               "More blunders than usual: take one extra half-second on captures and checks.", fmt="{:.1%}"),
        Metric("Blunder rate under 10s", w["low_clock_blunder_rate"], b["low_clock_blunder_rate"], False,
               "Low-clock accuracy slipped: keep pieces protected so fast moves stay safe.", fmt="{:.1%}"),
        Metric("Seconds used on first 10 moves", w["opening_time_used"], b["opening_time_used"], False,
               f"Openings got slower: aim to finish your first 10 moves in under {SLOW_OPENING_SECONDS}s.",
               fmt="{:.1f}s"),
        Metric(f"Games with a slow opening (>= {SLOW_OPENING_SECONDS}s)", w["slow_opening_share"],
               b["slow_opening_share"], False,
               "Drill your main lines so the first 10 moves are automatic."),
        Metric("Win rate in the first game of a session", w["first_game_win_rate"],
               b["first_game_win_rate"], True,
               "Cold starts cost you: play a warm-up game or puzzle rush before rated games."),
    ]
    info = {"week_start": week_start, "week_end": week_end, "week_games": len(week),
            "baseline_games": len(base)}
    return week, metrics, info


def standing_findings(games: pd.DataFrame) -> list[str]:
    """Headline patterns from my whole history, computed fresh each time (never hard-coded)."""
    slow = games["opening_time_used"] >= SLOW_OPENING_SECONDS
    first = games["game_index_in_session"] == 1
    after2 = games["loss_streak"] >= 2
    return [
        f"Openings: first 10 moves under {SLOW_OPENING_SECONDS}s -> {games.loc[~slow, 'won'].mean():.0%} wins; "
        f"slower -> {games.loc[slow, 'won'].mean():.0%} ({int(slow.sum())} games).",
        f"Warm-up: first game of a session -> {games.loc[first, 'won'].mean():.0%} wins; "
        f"later games -> {games.loc[~first, 'won'].mean():.0%}.",
        f"Tilt: after 2+ straight losses -> {games.loc[after2, 'won'].mean():.0%} wins; "
        f"otherwise -> {games.loc[~after2, 'won'].mean():.0%}.",
    ]
