"""Sessions, history-only tilt features, tilt patterns and stop-rule back-tests.

A session is a run of games where each game starts less than SESSION_GAP_MINUTES after the
previous one ended. Every feature for a game is built ONLY from earlier games in the same
session (plus pre-game facts such as the rating gap), so it could be known before the game starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_GAP_MINUTES = 30

# Per-game summaries carried forward as "previous game" information
HISTORY_COLS = ["won", "lost", "lost_on_time", "blunder_rate", "mean_time_spent", "opening_time_used"]

TILT_FEATURES = [
    "game_index_in_session", "loss_streak", "win_streak", "session_wins_before",
    "session_losses_before", "prev_won", "prev_lost_on_time", "prev_blunder_rate",
    "roll3_blunder_rate", "prev_mean_time_spent", "roll3_mean_time_spent",
    "prev_opening_time_used", "minutes_into_session", "elo_change_in_session",
    "elo_diff", "is_white",
]


def assign_sessions(games: pd.DataFrame, gap_minutes: int = SESSION_GAP_MINUTES) -> pd.DataFrame:
    """games needs utc_datetime (start) and end_datetime. Adds session_id and game_index_in_session."""
    df = games.copy()
    df["start"] = pd.to_datetime(df["utc_datetime"], utc=True, format="ISO8601")
    df["end"] = pd.to_datetime(df["end_datetime"], utc=True, format="ISO8601")
    df = df.sort_values("start").reset_index(drop=True)
    gap = (df["start"] - df["end"].shift()).dt.total_seconds() / 60
    df["session_id"] = (gap.isna() | (gap >= gap_minutes)).cumsum() - 1
    df["game_index_in_session"] = df.groupby("session_id").cumcount() + 1
    return df


def _streak(flags: pd.Series) -> pd.Series:
    """Length of the run of 1s ending at each position (inclusive)."""
    run = flags.groupby((flags == 0).cumsum()).cumsum()
    return run.where(flags == 1, 0)


def add_tilt_features(df: pd.DataFrame) -> pd.DataFrame:
    """df from assign_sessions joined with per-game summaries (game_features.csv columns)."""
    df = df.copy()
    df["won"] = (df["result"] == "win").astype(int)
    df["lost"] = (df["result"] == "loss").astype(int)
    df["is_white"] = (df["colour"] == "white").astype(int)
    g = df.groupby("session_id", group_keys=False)

    # Everything below is shifted by one game inside the session: game i only sees games < i.
    df["loss_streak"] = g["lost"].apply(lambda s: _streak(s).shift(fill_value=0))
    df["win_streak"] = g["won"].apply(lambda s: _streak(s).shift(fill_value=0))
    df["session_wins_before"] = g["won"].apply(lambda s: s.cumsum().shift(fill_value=0))
    df["session_losses_before"] = g["lost"].apply(lambda s: s.cumsum().shift(fill_value=0))
    for col in HISTORY_COLS:
        df[f"prev_{col}"] = g[col].shift()
    for col in ["blunder_rate", "mean_time_spent"]:
        df[f"roll3_{col}"] = g[col].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    df["minutes_into_session"] = (df["start"] - g["start"].transform("first")).dt.total_seconds() / 60
    df["elo_change_in_session"] = df["my_elo"] - g["my_elo"].transform("first")

    # First game of a session has no history in that session: use 0 for the result flags and,
    # for the numeric summaries, my average over all EARLIER games (never the full-data mean).
    for col in ["won", "lost", "lost_on_time"]:
        df[f"prev_{col}"] = df[f"prev_{col}"].fillna(0)
    for col in ["blunder_rate", "mean_time_spent", "opening_time_used"]:
        past_mean = df[col].shift().expanding().mean().fillna(0)
        df[f"prev_{col}"] = df[f"prev_{col}"].fillna(past_mean)
        if f"roll3_{col}" in df:
            df[f"roll3_{col}"] = df[f"roll3_{col}"].fillna(past_mean)
    return df


# ---------------------------------------------------------------- descriptive tilt patterns

def win_rate_by(df: pd.DataFrame, col: str, bins: list, labels: list[str]) -> pd.DataFrame:
    b = pd.cut(df[col], bins, labels=labels, right=False)
    return df.groupby(b, observed=False).agg(
        games=("won", "size"), win_rate=("won", "mean"), lost_on_time=("lost_on_time", "mean"),
        blunder_rate=("blunder_rate", "mean"), mean_time_spent=("mean_time_spent", "mean"),
    )


# ---------------------------------------------------------------- stop rules

def stop_rules() -> dict[str, callable]:
    """Each rule returns True for games that would still be played (kept)."""
    rules = {}
    for n in (1, 2, 3, 4):
        rules[f"stop after {n} straight loss{'es' if n > 1 else ''}"] = (
            lambda d, n=n: d["loss_streak"] < n)
    for n in (5, 10, 15, 20, 30):
        rules[f"stop after {n} games"] = lambda d, n=n: d["game_index_in_session"] <= n
    for k in (2, 3, 4):
        rules[f"stop when {k} losses down"] = (
            lambda d, k=k: (d["session_losses_before"] - d["session_wins_before"]) < k)
    return rules


def backtest_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Kept vs dropped win rate for every rule. NOTE: kept games are assumed to be unchanged
    by stopping, i.e. the games you'd still play would have gone the same way."""
    base = df["won"].mean()
    rows = []
    for name, rule in stop_rules().items():
        kept = rule(df)
        rows.append({
            "rule": name,
            "kept_games": int(kept.sum()),
            "kept_share": kept.mean(),
            "kept_win_rate": df.loc[kept, "won"].mean(),
            "dropped_win_rate": df.loc[~kept, "won"].mean() if (~kept).any() else np.nan,
            "gain_points": (df.loc[kept, "won"].mean() - base) * 100,
        })
    return pd.DataFrame(rows).set_index("rule")


def choose_rule(train: pd.DataFrame, min_kept_share: float = 0.5) -> str:
    """Best rule on older sessions, ignoring rules that would throw away most games."""
    bt = backtest_rules(train)
    return bt[bt["kept_share"] >= min_kept_share]["gain_points"].idxmax()


def split_sessions(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Newest test_frac of SESSIONS (by start time) for testing; sessions are never split."""
    order = df.groupby("session_id")["start"].min().sort_values()
    n_test = max(1, round(len(order) * test_frac))
    test_ids = set(order.index[-n_test:])
    is_test = df["session_id"].isin(test_ids)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
