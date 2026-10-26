"""Step 4a: predict my blunders (is_blunder_wc) from clock and position.

Usage (from the project root):
    python -m tilt.blunder_model

Reads:
    data/processed/moves_labelled.csv, data/processed/games.csv
Writes:
    models/blunder_xgb.json
    reports/blunder_model/metrics.csv              test metrics for every model
    reports/blunder_model/feature_importance.csv   permutation importance (XGBoost, test set)
    reports/blunder_model/clock_effect.csv         model risk by clock, other factors held fixed
    reports/blunder_model/summary.txt              everything printed, incl. plain-language advice

Method:
- Time-based split by game: newest 20% of games = test. The newest 20% of the remaining
  games = validation, used only to pick each model's decision threshold (best F1).
  Models are then refit on train + validation and scored once on test.
- Class imbalance: class_weight="balanced" (LR, RF) and scale_pos_weight (XGBoost).
- Judged by ROC-AUC, PR-AUC, recall, precision and F1, never accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tilt.config import (
    GAMES_CSV_PATH, LABELLED_MOVES_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR,
)
from tilt.features import FEATURES, PHASES, TARGET, build_features, time_split

GOAL_AUC = 0.75
# Representative clock values (seconds) for the handoff's buckets
CLOCK_POINTS = {"<3s": 2.0, "3-5s": 4.0, "5-10s": 7.5, "10-30s": 20.0, "30s+": 45.0}
REFERENCE_BUCKET = "30s+"


def make_models(pos_weight: float) -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="auc",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def best_f1_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": threshold,
    }


def clock_effect(model, X: pd.DataFrame, phases: pd.Series) -> pd.DataFrame:
    """Partial dependence on the clock: set every test move's clock to each bucket value
    (keeping clock_ratio consistent) and average the predicted risk, overall and per phase.
    Risks are relative: class weighting inflates the absolute probabilities.
    """
    rows = []
    for bucket, secs in CLOCK_POINTS.items():
        Xc = X.copy()
        Xc["clock_before"] = secs
        Xc["clock_ratio"] = secs / X["opp_clock"].clip(lower=0.1)
        proba = model.predict_proba(Xc)[:, 1]
        row = {"bucket": bucket, "seconds": secs, "all": proba.mean()}
        for p in PHASES:
            mask = (phases == p).to_numpy()
            row[p] = proba[mask].mean() if mask.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("bucket")
    ratios = df[["all", *PHASES]] / df.loc[REFERENCE_BUCKET, ["all", *PHASES]]
    return df.join(ratios.add_suffix("_vs_30s"))


def level_position_rates(test: pd.DataFrame) -> pd.Series:
    """Observed blunder rate by clock bucket, only in level positions (|eval_before| < 2)."""
    level = test[test["abs_eval_before"] < 2]
    buckets = pd.cut(level["clock_before"], [0, 3, 5, 10, 30, 61], right=False,
                     labels=list(CLOCK_POINTS))
    return level.groupby(buckets, observed=False)[TARGET].agg(["mean", "size"])


def advice(effect: pd.DataFrame, importance: pd.DataFrame) -> list[str]:
    """Turn the clock effect and importances into plain-language sentences."""
    lines = []
    for col, label in [("all", "overall"), *[(p, f"in the {p}") for p in PHASES]]:
        r = effect.loc["<3s", f"{col}_vs_30s"]
        if np.isnan(r):
            continue
        doubled = [b for b in CLOCK_POINTS if effect.loc[b, f"{col}_vs_30s"] >= 2]
        msg = f"With other factors held fixed, {label} your blunder risk below 3s is {r:.1f}x your risk with 30s+ left"
        msg += (f"; it at least doubles from the {doubled[-1]} bucket down." if doubled
                else "; no clock bucket doubles it.")
        lines.append(msg)
    top = importance.head(3)["feature"].tolist()
    lines.append(f"The three factors the model leans on most: {', '.join(top)}.")
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--moves", type=Path, default=LABELLED_MOVES_PATH)
    p.add_argument("--games", type=Path, default=GAMES_CSV_PATH)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR / "blunder_model")
    p.add_argument("--models", type=Path, default=MODELS_DIR)
    args = p.parse_args(argv)

    df = build_features(pd.read_csv(args.moves), pd.read_csv(args.games))
    train_all, test = time_split(df, 0.2)
    train, val = time_split(train_all, 0.2)

    out: list[str] = ["=== Blunder model (target: is_blunder_wc) ==="]
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        out.append(
            f"{name:<11} games: {part['game_id'].nunique():>4}  moves: {len(part):>6}  "
            f"blunder rate: {part[TARGET].mean():.2%}  "
            f"({part['utc_datetime'].min()[:10]} to {part['utc_datetime'].max()[:10]})"
        )

    pos_weight = (train[TARGET] == 0).sum() / max(1, train[TARGET].sum())
    metrics, fitted = {}, {}
    for name, model in make_models(pos_weight).items():
        model.fit(train[FEATURES], train[TARGET])
        threshold = best_f1_threshold(val[TARGET].to_numpy(), model.predict_proba(val[FEATURES])[:, 1])
        final = make_models(pos_weight)[name].fit(train_all[FEATURES], train_all[TARGET])
        metrics[name] = evaluate(test[TARGET].to_numpy(), final.predict_proba(test[FEATURES])[:, 1], threshold)
        fitted[name] = final
    metrics_df = pd.DataFrame(metrics).T.round(3)
    out += ["", "Test metrics (newest 20% of games):", metrics_df.to_string()]
    best = metrics_df["roc_auc"].idxmax()
    best_auc = metrics_df.loc[best, "roc_auc"]
    out.append(f"Goal ROC-AUC >= {GOAL_AUC}: {'MET' if best_auc >= GOAL_AUC else 'NOT met'} "
               f"(best: {best} {best_auc:.3f})")

    xgb = fitted["xgboost"]
    perm = permutation_importance(xgb, test[FEATURES], test[TARGET], scoring="roc_auc",
                                  n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURES, "auc_drop": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("auc_drop", ascending=False).reset_index(drop=True))
    out += ["", "Permutation importance (drop in test ROC-AUC when a feature is shuffled):",
            importance.round(4).to_string(index=False)]

    effect = clock_effect(xgb, test[FEATURES], test["phase"])
    out += ["", "Clock effect, other factors held fixed (risk relative to 30s+):",
            effect.filter(like="_vs_30s").round(2).to_string()]
    level = level_position_rates(test)
    out += ["", "Observed test blunder rate in level positions (|eval| < 2) by clock:",
            level.assign(mean=level["mean"].map("{:.2%}".format)).to_string()]
    out += ["", "Plain-language findings:", *(f"- {s}" for s in advice(effect, importance))]

    args.reports.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.reports / "metrics.csv")
    importance.to_csv(args.reports / "feature_importance.csv", index=False)
    effect.to_csv(args.reports / "clock_effect.csv")
    xgb.save_model(args.models / "blunder_xgb.json")
    text = "\n".join(out)
    (args.reports / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved reports to: {args.reports}")


if __name__ == "__main__":
    main()
