"""Tests for the feature table, time split and blunder model, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from tilt import blunder_model
from tilt.features import FEATURES, FORBIDDEN, TARGET, build_features, time_split


def synthetic(n_games=200, moves_per_game=30, seed=42):
    """Fake labelled moves where blunder risk rises sharply as the clock runs down."""
    rng = np.random.default_rng(seed)
    games = pd.DataFrame({
        "game_id": np.arange(n_games),
        "utc_datetime": pd.date_range("2026-09-01", periods=n_games, freq="h", tz="UTC").map(
            lambda t: t.isoformat()),
        "my_elo": 1500, "elo_diff": rng.integers(-100, 100, n_games),
    })
    n = n_games * moves_per_game
    clock = rng.uniform(0.2, 60, n)
    moves = pd.DataFrame({
        "game_id": np.repeat(games["game_id"], moves_per_game).to_numpy(),
        "ply": np.tile(np.arange(1, moves_per_game + 1), n_games),
        "is_me": True, "colour": "white",
        "phase": rng.choice(["opening", "middlegame", "endgame"], n),
        "move_number": rng.integers(1, 60, n), "clock_before": clock,
        "time_spent": rng.uniform(0, 3, n), "opp_clock": rng.uniform(0.2, 60, n),
        "clock_ratio": np.nan, "material_balance": rng.integers(-5, 5, n),
        "legal_moves": rng.integers(1, 40, n), "in_check": False, "is_capture": False,
        "eval_before": rng.normal(0, 2, n), "eval_drop": 0.0, "is_blunder": 0,
    })
    moves[TARGET] = (rng.random(n) < 0.05 + 0.5 * np.exp(-clock / 5)).astype(int)
    return moves, games


def test_no_leaky_features():
    assert not FORBIDDEN & set(FEATURES)


def test_time_split_puts_newest_games_in_test():
    df = build_features(*synthetic())
    train, test = time_split(df, 0.2)
    assert test["game_id"].nunique() == 40
    assert train["utc_datetime"].max() < test["utc_datetime"].min()
    assert not set(train["game_id"]) & set(test["game_id"])


def test_end_to_end_finds_clock_effect(tmp_path):
    moves, games = synthetic()
    moves.to_csv(tmp_path / "moves.csv", index=False)
    games.to_csv(tmp_path / "games.csv", index=False)
    blunder_model.main([
        "--moves", str(tmp_path / "moves.csv"), "--games", str(tmp_path / "games.csv"),
        "--reports", str(tmp_path / "reports"), "--models", str(tmp_path / "models"),
    ])
    metrics = pd.read_csv(tmp_path / "reports" / "metrics.csv", index_col=0)
    assert metrics.loc["xgboost", "roc_auc"] > 0.6  # synthetic ceiling is about 0.67
    effect = pd.read_csv(tmp_path / "reports" / "clock_effect.csv", index_col=0)
    assert effect.loc["<3s", "all_vs_30s"] > 2
    importance = pd.read_csv(tmp_path / "reports" / "feature_importance.csv")
    assert importance.iloc[0]["feature"] in {"clock_before", "clock_ratio"}
    assert (tmp_path / "models" / "blunder_xgb.json").exists()
