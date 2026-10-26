"""Per-game summaries of my play, built from the labelled move table.

STYLE_FEATURES describe HOW a game was played (time use, speed, game shape) and are used
for clustering. They contain no outcome information. Quality and outcome columns
(blunder_rate, result, lost_on_time) are kept alongside for describing clusters and for
the later session/tilt model, never as clustering inputs.
"""

from __future__ import annotations

import pandas as pd

STYLE_FEATURES = [
    "n_my_moves",          # game length
    "mean_time_spent",     # average seconds per move
    "fast_move_share",     # share of moves played in under 0.5s
    "opening_time_used",   # seconds used on my first 10 moves
    "mean_clock_ratio",    # my clock / opponent's clock, averaged over the game
    "endgame_share",       # share of my moves in the endgame
    "capture_share",       # share of my moves that capture
    "mean_legal_moves",    # average complexity of the positions I faced
]
FAST_MOVE_SECONDS = 0.5
OPENING_MOVES = 10
TIMEOUT_CODES = {"timeout"}


def build_game_features(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    mine = moves[moves["is_me"].astype(bool)].sort_values(["game_id", "ply"]).copy()
    mine["is_capture"] = mine["is_capture"].astype(bool)
    g = mine.groupby("game_id")
    feats = pd.DataFrame({
        "n_my_moves": g.size(),
        "mean_time_spent": g["time_spent"].mean(),
        "fast_move_share": g["time_spent"].apply(lambda s: (s < FAST_MOVE_SECONDS).mean()),
        "opening_time_used": g["time_spent"].apply(lambda s: s.head(OPENING_MOVES).sum()),
        "mean_clock_ratio": g["clock_ratio"].apply(lambda s: s.clip(upper=5).mean()),
        "endgame_share": g["phase"].apply(lambda s: (s == "endgame").mean()),
        "capture_share": g["is_capture"].mean(),
        "mean_legal_moves": g["legal_moves"].mean(),
        # --- not for clustering: quality / outcome, used to describe clusters and for the tilt model
        "blunder_rate": g["is_blunder_wc"].mean() if "is_blunder_wc" in mine else pd.NA,
    }).reset_index()

    context = games[["game_id", "utc_datetime", "colour", "my_elo", "elo_diff",
                     "opening_family", "result", "my_result_code"]]
    out = context.merge(feats, on="game_id", how="inner", validate="one_to_one")
    out["lost_on_time"] = out["my_result_code"].isin(TIMEOUT_CODES).astype(int)
    return out.sort_values("utc_datetime").reset_index(drop=True)
