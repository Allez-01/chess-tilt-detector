"""Tests for sessions, history-only tilt features, stop rules and the tilt CLI."""

import numpy as np
import pandas as pd
import pytest

from tilt import tilt_model
from tilt.sessions import TILT_FEATURES, add_tilt_features, assign_sessions, backtest_rules


def synthetic(n_sessions=60, per_session=10, seed=42):
    """Sessions of games 5 min apart, 3 h between sessions. Win chance drops after 2 losses."""
    rng = np.random.default_rng(seed)
    rows, t, gid = [], pd.Timestamp("2026-09-01", tz="UTC"), 0
    for _ in range(n_sessions):
        streak = 0
        for _ in range(per_session):
            won = rng.random() < (0.6 if streak < 2 else 0.2)
            streak = 0 if won else streak + 1
            rows.append({
                "game_id": gid, "utc_datetime": t.isoformat(),
                "end_datetime": (t + pd.Timedelta(minutes=2)).isoformat(),
                "result": "win" if won else "loss", "my_result_code": "win" if won else "timeout",
                "colour": "white" if gid % 2 else "black", "my_elo": 1500 + rng.integers(-20, 20),
                "elo_diff": int(rng.integers(-50, 50)), "blunder_rate": rng.uniform(0, 0.2),
                "mean_time_spent": rng.uniform(1, 2), "opening_time_used": rng.uniform(5, 20),
                "lost_on_time": int(not won),
            })
            gid += 1
            t += pd.Timedelta(minutes=5)
        t += pd.Timedelta(hours=3)
    return pd.DataFrame(rows)


def test_sessions_split_on_gaps():
    df = assign_sessions(synthetic(n_sessions=3, per_session=4))
    assert df["session_id"].nunique() == 3
    assert df["game_index_in_session"].tolist() == [1, 2, 3, 4] * 3


def test_streaks_use_only_earlier_games():
    games = synthetic(n_sessions=1, per_session=5)
    games["result"] = ["loss", "loss", "win", "loss", "win"]
    df = add_tilt_features(assign_sessions(games))
    assert df["loss_streak"].tolist() == [0, 1, 2, 0, 1]
    assert df["session_losses_before"].tolist() == [0, 1, 2, 2, 3]


def test_no_lookahead():
    games = synthetic(n_sessions=2, per_session=6)
    a = add_tilt_features(assign_sessions(games))
    flipped = games.copy()
    flipped.loc[3, "result"] = "win" if games.loc[3, "result"] == "loss" else "loss"
    flipped.loc[3, "blunder_rate"] = 0.99
    b = add_tilt_features(assign_sessions(flipped))
    pd.testing.assert_frame_equal(a.loc[:3, TILT_FEATURES], b.loc[:3, TILT_FEATURES])


def test_stop_rule_finds_planted_tilt():
    df = add_tilt_features(assign_sessions(synthetic()))
    bt = backtest_rules(df)
    assert bt.loc["stop after 2 straight losses", "gain_points"] > 5


def test_cli_end_to_end(tmp_path):
    games = synthetic()
    games.drop(columns=["end_datetime"]).to_csv(tmp_path / "gf.csv", index=False)
    games[["game_id", "end_datetime"]].to_csv(tmp_path / "games.csv", index=False)
    pytest.importorskip("tensorflow")
    tilt_model.main(["--game-features", str(tmp_path / "gf.csv"), "--games", str(tmp_path / "games.csv"),
                     "--out", str(tmp_path / "sg.csv"), "--reports", str(tmp_path / "rep")])
    metrics = pd.read_csv(tmp_path / "rep" / "metrics.csv", index_col=0)
    assert {"baseline_elo_only", "logistic_tilt", "xgboost_tilt", "lstm"} <= set(metrics.index)
    assert metrics.loc["logistic_tilt", "roc_auc"] > metrics.loc["baseline_elo_only", "roc_auc"]
