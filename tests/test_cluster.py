"""Tests for per-game features and play-style clustering, on synthetic games with planted styles."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from tilt import cluster
from tilt.game_features import STYLE_FEATURES, build_game_features

STYLES = {  # time per move, moves per game, endgame share, capture share
    0: (0.4, 20, 0.0, 0.1),
    1: (1.8, 55, 0.6, 0.2),
    2: (1.0, 35, 0.2, 0.45),
}


def synthetic(n_games=150, seed=42):
    rng = np.random.default_rng(seed)
    style = np.arange(n_games) % 3
    move_rows, game_rows = [], []
    for gid, s in enumerate(style):
        t, n, endgame, captures = STYLES[s]
        n = int(n + rng.integers(-3, 4))
        spent = np.clip(rng.normal(t, 0.15, n), 0, None)
        clock = 60 - np.cumsum(spent)
        move_rows.append(pd.DataFrame({
            "game_id": gid, "ply": np.arange(1, n + 1) * 2, "is_me": True,
            "time_spent": spent, "clock_ratio": rng.normal(1, 0.05, n), "clock_before": clock,
            "phase": np.where(rng.random(n) < endgame, "endgame", "middlegame"),
            "is_capture": rng.random(n) < captures, "legal_moves": rng.integers(20, 35, n),
            "is_blunder_wc": (rng.random(n) < 0.1).astype(int),
        }))
        game_rows.append({
            "game_id": gid, "utc_datetime": f"2026-10-01T{gid % 24:02d}:00:00+00:00",
            "colour": "white", "my_elo": 1500, "elo_diff": 0, "opening_family": "x",
            "result": "win" if s == 0 else "loss", "my_result_code": "win" if s == 0 else "timeout",
        })
    return pd.concat(move_rows), pd.DataFrame(game_rows), style


def test_style_features_have_no_outcomes():
    assert not {"result", "my_result_code", "lost_on_time", "blunder_rate"} & set(STYLE_FEATURES)


def test_game_features_values():
    moves, games, _ = synthetic()
    df = build_game_features(moves, games)
    assert len(df) == len(games)
    assert df.loc[df["game_id"] == 0, "fast_move_share"].item() > 0.5
    assert df["lost_on_time"].sum() == 100


def test_clustering_recovers_planted_styles(tmp_path):
    moves, games, style = synthetic()
    moves.to_csv(tmp_path / "moves.csv", index=False)
    games.to_csv(tmp_path / "games.csv", index=False)
    cluster.main(["--moves", str(tmp_path / "moves.csv"), "--games", str(tmp_path / "games.csv"),
                  "--features-out", str(tmp_path / "gf.csv"), "--reports", str(tmp_path / "rep")])
    out = pd.read_csv(tmp_path / "gf.csv").sort_values("game_id")
    assert out["cluster"].nunique() == 3
    assert adjusted_rand_score(style, out["cluster"]) > 0.9
    prof = pd.read_csv(tmp_path / "rep" / "profiles.csv", index_col=0)
    best = prof.iloc[0]  # sorted by win rate: the planted fast/short style
    assert best["win_rate"] == 1.0
    assert any(w in best["name"] for w in ("quick moves", "short games", "many premoves", "fast opening"))
