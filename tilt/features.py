"""Move-level feature table and time-based split for the blunder model.

Only information available before or at the moment of the move is used. Outcome-revealing
fields (result, termination, accuracies) and label-building fields (eval_after, eval_drop,
wc_drop) are never features; build_features() checks this.
"""

from __future__ import annotations

import pandas as pd

TARGET = "is_blunder_wc"

NUMERIC = [
    "move_number", "clock_before", "time_spent", "opp_clock", "clock_ratio",
    "material_balance", "legal_moves", "eval_before", "abs_eval_before", "my_elo", "elo_diff",
]
BOOLEAN = ["in_check", "is_capture", "is_white"]
PHASES = ["opening", "middlegame", "endgame"]
FEATURES = NUMERIC + BOOLEAN + [f"phase_{p}" for p in PHASES]

FORBIDDEN = {
    "result", "my_result_code", "opp_result_code", "termination", "my_accuracy", "opp_accuracy",
    "eval_after", "eval_drop", "wc_drop", "is_blunder", "is_blunder_wc",
    "my_rating_after", "opp_rating_after",
}


def build_features(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """My moves only, with features, target, game_id, phase and the game's start time."""
    assert not FORBIDDEN & set(FEATURES), "outcome or label column used as a feature"

    mine = moves[moves["is_me"].astype(bool)].copy()
    game_cols = games[["game_id", "utc_datetime", "my_elo", "elo_diff"]]
    df = mine.merge(game_cols, on="game_id", how="left", validate="many_to_one")

    df["abs_eval_before"] = df["eval_before"].abs()
    df["is_white"] = df["colour"].eq("white")
    for p in PHASES:
        df[f"phase_{p}"] = df["phase"].eq(p).astype(int)
    for col in BOOLEAN:
        df[col] = df[col].astype(int)
    df["clock_ratio"] = df["clock_ratio"].fillna(df["clock_before"] / df["opp_clock"].clip(lower=0.1))

    keep = ["game_id", "utc_datetime", "ply", "phase"] + FEATURES + [TARGET]
    return df[keep].dropna(subset=["eval_before", TARGET]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by GAME start time: the newest test_frac of games go to test, never shuffled."""
    order = df.groupby("game_id")["utc_datetime"].first().sort_values()
    n_test = max(1, round(len(order) * test_frac))
    test_games = set(order.index[-n_test:])
    is_test = df["game_id"].isin(test_games)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
