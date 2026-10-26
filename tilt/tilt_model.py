"""Step 4c: sessions, tilt patterns, stop-rule back-test and next-game models.

Usage (from the project root, after step 4b):
    python -m tilt.tilt_model            # everything, including the LSTM
    python -m tilt.tilt_model --no-lstm  # skip TensorFlow

Reads:
    data/processed/game_features.csv (step 4b), data/processed/games.csv
Writes:
    data/processed/session_games.csv   games with session_id and history-only tilt features
    reports/tilt/patterns.csv, rules.csv, metrics.csv, coefficients.csv, summary.txt

Method:
- Sessions: gaps under 30 minutes. Split by SESSION start time: newest 20% of sessions = test.
- Stop rules are chosen on the older sessions only, then scored once on the newest sessions.
  Goal: kept games win at least 5 points more often than all games.
- Next-game models predict "this game is a win" from the earlier games in the session only.
  A rating-gap-only model is the baseline: tilt features must beat it to be worth anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tilt.config import (
    GAME_FEATURES_PATH, GAMES_CSV_PATH, RANDOM_STATE, REPORTS_DIR, SESSION_GAMES_PATH,
)
from tilt.sessions import (
    HISTORY_COLS, TILT_FEATURES, add_tilt_features, assign_sessions, backtest_rules,
    choose_rule, split_sessions, stop_rules, win_rate_by,
)

GOAL_GAIN_POINTS = 5.0
LSTM_STEPS = 5  # how many previous games in the session the LSTM sees


# ---------------------------------------------------------------- tabular models

def tabular_models() -> dict:
    return {
        "baseline_elo_only": (["elo_diff"], make_pipeline(
            StandardScaler(), LogisticRegression(random_state=RANDOM_STATE))),
        "logistic_tilt": (TILT_FEATURES, make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_STATE))),
        "xgboost_tilt": (TILT_FEATURES, XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)),
    }


def score(y, proba) -> dict:
    return {"roc_auc": roc_auc_score(y, proba), "brier": brier_score_loss(y, proba),
            "accuracy": accuracy_score(y, proba >= 0.5)}


# ---------------------------------------------------------------- LSTM

def build_sequences(df: pd.DataFrame, means: pd.Series, stds: pd.Series):
    """For each game: the previous LSTM_STEPS games of its session (oldest first, left-padded
    with zeros), preceded by a 'session start' step so no sequence is empty. Plus static inputs."""
    n_feat = len(HISTORY_COLS) + 1  # + start flag
    X = np.zeros((len(df), LSTM_STEPS + 1, n_feat), dtype="float32")
    hist = ((df[HISTORY_COLS] - means) / stds).to_numpy(dtype="float32")
    for _, sess in df.groupby("session_id", sort=False):
        idx = sess.index.to_numpy()
        for j, row in enumerate(idx):
            prev = idx[max(0, j - LSTM_STEPS):j]
            start = LSTM_STEPS - len(prev)
            X[row, start, -1] = 1.0  # start token right before the real history
            if len(prev):
                X[row, start + 1:, :-1] = hist[prev]
    static = df[["elo_diff", "game_index_in_session"]].to_numpy(dtype="float32")
    return X, static


def run_lstm(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    import tensorflow as tf  # imported here so the rest works without TensorFlow

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    means, stds = train[HISTORY_COLS].mean(), train[HISTORY_COLS].std().replace(0, 1)
    fit, val = split_sessions(train, 0.2)
    s_mean = fit[["elo_diff", "game_index_in_session"]].mean().to_numpy("float32")
    s_std = fit[["elo_diff", "game_index_in_session"]].std().replace(0, 1).to_numpy("float32")

    def inputs(d):
        X, s = build_sequences(d.reset_index(drop=True), means, stds)
        return X, (s - s_mean) / s_std

    seq_in = tf.keras.Input(shape=(LSTM_STEPS + 1, len(HISTORY_COLS) + 1), name="history")
    static_in = tf.keras.Input(shape=(2,), name="static")
    h = tf.keras.layers.Masking(mask_value=0.0)(seq_in)
    h = tf.keras.layers.LSTM(16)(h)
    h = tf.keras.layers.Concatenate()([h, static_in])
    h = tf.keras.layers.Dense(16, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.2)(h)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(h)
    model = tf.keras.Model([seq_in, static_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])

    (Xf, sf), (Xv, sv), (Xt, st) = inputs(fit), inputs(val), inputs(test)
    model.fit([Xf, sf], fit["won"].to_numpy(), validation_data=([Xv, sv], val["won"].to_numpy()),
              epochs=60, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                          patience=8, restore_best_weights=True)])
    return score(test["won"].to_numpy(), model.predict([Xt, st], verbose=0).ravel())


# ---------------------------------------------------------------- report

def fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    return out


def findings(streak: pd.DataFrame, chosen: str, test_bt: pd.DataFrame, test_base: float,
             metrics: pd.DataFrame) -> list[str]:
    lines = []
    w0, w2 = streak.loc["0", "win_rate"], streak.loc["2", "win_rate"]
    if streak.loc["2", "games"] >= 20:
        lines.append(f"After 2 straight losses in a session you win {w2:.0%} of the next game, "
                     f"vs {w0:.0%} when your last game wasn't a loss.")
    r = test_bt.loc[chosen]
    gain = (r["kept_win_rate"] - test_base) * 100
    lines.append(
        f"Best stop rule on your older sessions: '{chosen}'. On your newest sessions it keeps "
        f"{r['kept_share']:.0%} of games, which win {r['kept_win_rate']:.0%} vs {test_base:.0%} overall "
        f"({gain:+.1f} points; goal +{GOAL_GAIN_POINTS:.0f}: {'MET' if gain >= GOAL_GAIN_POINTS else 'NOT met'})."
    )
    base = metrics.loc["baseline_elo_only", "roc_auc"]
    best = metrics["roc_auc"].drop("baseline_elo_only").idxmax()
    lines.append(
        f"Predicting the next game: rating gap alone gives ROC-AUC {base:.3f}; the best tilt model "
        f"({best}) gives {metrics.loc[best, 'roc_auc']:.3f}, so session history adds "
        f"{metrics.loc[best, 'roc_auc'] - base:+.3f}."
    )
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--game-features", type=Path, default=GAME_FEATURES_PATH)
    p.add_argument("--games", type=Path, default=GAMES_CSV_PATH)
    p.add_argument("--out", type=Path, default=SESSION_GAMES_PATH)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR / "tilt")
    args = p.parse_args(argv)

    gf = pd.read_csv(args.game_features)
    ends = pd.read_csv(args.games, usecols=["game_id", "end_datetime"])
    df = add_tilt_features(assign_sessions(gf.merge(ends, on="game_id", how="left")))
    sizes = df.groupby("session_id").size()

    out = [
        f"=== Sessions and tilt ({len(df)} games) ===",
        f"Sessions: {len(sizes)}  games per session: median {sizes.median():.0f}, "
        f"mean {sizes.mean():.1f}, max {sizes.max()}",
        f"Overall win rate: {df['won'].mean():.1%}",
    ]

    streak = win_rate_by(df, "loss_streak", [0, 1, 2, 3, np.inf], ["0", "1", "2", "3+"])
    index = win_rate_by(df, "game_index_in_session", [1, 2, 6, 11, 21, np.inf],
                        ["1st", "2-5", "6-10", "11-20", "21+"])
    pct = ["win_rate", "lost_on_time", "blunder_rate"]
    out += ["", "By losses in a row before the game:", fmt_pct(streak, pct).round(2).to_string(),
            "", "By game number in the session:", fmt_pct(index, pct).round(2).to_string()]

    train, test = split_sessions(df, 0.2)
    chosen = choose_rule(train)
    all_bt = backtest_rules(df)
    test_bt = backtest_rules(test)
    out += ["", "Stop rules on all games (kept = games you'd still have played):",
            fmt_pct(all_bt, ["kept_share", "kept_win_rate", "dropped_win_rate"])
            .round(1).to_string(),
            f"Rule chosen on older {train['session_id'].nunique()} sessions: {chosen}",
            f"On newest {test['session_id'].nunique()} sessions ({len(test)} games): kept "
            f"{test_bt.loc[chosen, 'kept_games']} games, win rate {test_bt.loc[chosen, 'kept_win_rate']:.1%} "
            f"vs {test['won'].mean():.1%} for all their games"]

    metrics, fitted = {}, {}
    for name, (cols, model) in tabular_models().items():
        model.fit(train[cols], train["won"])
        metrics[name] = score(test["won"].to_numpy(), model.predict_proba(test[cols])[:, 1])
        fitted[name] = model
    if not args.no_lstm:
        try:
            metrics["lstm"] = run_lstm(train, test)
        except ImportError:
            out.append("TensorFlow not installed: LSTM skipped (pip install tensorflow).")
    metrics_df = pd.DataFrame(metrics).T
    if metrics_df.loc["baseline_elo_only", "roc_auc"] > 0.75:
        out.append("WARNING: the rating gap alone predicts results far too well; "
                   "check that ratings are pre-game (leakage).")
    coefs = pd.Series(fitted["logistic_tilt"][-1].coef_[0], index=TILT_FEATURES,
                      name="coef_per_sd").sort_values(key=abs, ascending=False)
    out += ["", f"Next-game win models (test = newest sessions, {len(test)} games):",
            metrics_df.round(3).to_string(),
            "", "Logistic tilt model coefficients (per standard deviation; + = more likely to win):",
            coefs.round(3).to_string(),
            "", "Plain-language findings:",
            *(f"- {s}" for s in findings(streak, chosen, test_bt, test["won"].mean(), metrics_df))]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["start", "end"]).to_csv(args.out, index=False)
    args.reports.mkdir(parents=True, exist_ok=True)
    pd.concat({"loss_streak": streak, "game_index": index}).to_csv(args.reports / "patterns.csv")
    all_bt.join(test_bt.add_suffix("_test")).to_csv(args.reports / "rules.csv")
    metrics_df.to_csv(args.reports / "metrics.csv")
    coefs.to_csv(args.reports / "coefficients.csv")
    text = "\n".join(out)
    (args.reports / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved reports to: {args.reports}")


if __name__ == "__main__":
    main()
