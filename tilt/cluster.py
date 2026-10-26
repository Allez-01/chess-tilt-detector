"""Step 4b: group my games into play-styles with k-means.

Usage (from the project root):
    python -m tilt.cluster            # picks k in 3..5 by silhouette score
    python -m tilt.cluster --k 4      # force a k

Reads:
    data/processed/moves_labelled.csv, data/processed/games.csv
Writes:
    data/processed/game_features.csv    per-game style features + cluster (reused by the tilt model)
    reports/clustering/silhouette.csv
    reports/clustering/profiles.csv     per-cluster style means, win rate, time losses, blunder rate
    reports/clustering/summary.txt

Clustering uses STYLE_FEATURES only (standardised). Results, time losses and blunder rate are
attached afterwards to describe each cluster; they never influence which cluster a game joins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from tilt.config import (
    GAME_FEATURES_PATH, GAMES_CSV_PATH, LABELLED_MOVES_PATH, RANDOM_STATE, REPORTS_DIR,
)
from tilt.game_features import STYLE_FEATURES, build_game_features

K_RANGE = range(3, 6)

# Words used to name a cluster from its most distinctive features (high, low)
PHRASES = {
    "n_my_moves": ("long games", "short games"),
    "mean_time_spent": ("slow moves", "quick moves"),
    "fast_move_share": ("many premoves", "few premoves"),
    "opening_time_used": ("slow opening", "fast opening"),
    "mean_clock_ratio": ("ahead on the clock", "behind on the clock"),
    "endgame_share": ("endgame-heavy", "few endgames"),
    "capture_share": ("trade-heavy", "few trades"),
    "mean_legal_moves": ("complex positions", "simple positions"),
}


def choose_k(X, k_range=K_RANGE) -> tuple[int, pd.Series]:
    scores = {
        k: silhouette_score(X, KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(X))
        for k in k_range
    }
    s = pd.Series(scores, name="silhouette").rename_axis("k")
    return int(s.idxmax()), s


def name_clusters(centres_z: pd.DataFrame, n_terms: int = 2) -> dict[int, str]:
    names = {}
    for c, row in centres_z.iterrows():
        top = row.abs().sort_values(ascending=False).index[:n_terms]
        names[c] = ", ".join(PHRASES[f][0 if row[f] > 0 else 1] for f in top)
    return names


def profile(df: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    g = df.groupby("cluster")
    prof = pd.DataFrame({
        "name": pd.Series(names),
        "games": g.size(),
        "win_rate": g["result"].apply(lambda s: (s == "win").mean()),
        "loss_rate": g["result"].apply(lambda s: (s == "loss").mean()),
        "lost_on_time": g["lost_on_time"].mean(),
        "blunder_rate": g["blunder_rate"].mean(),
        "mean_elo_diff": g["elo_diff"].mean(),
    })
    return prof.join(g[STYLE_FEATURES].mean()).sort_values("win_rate", ascending=False)


def findings(prof: pd.DataFrame, overall_win: float) -> list[str]:
    best, worst = prof.iloc[0], prof.iloc[-1]
    lines = [
        f"Your best style is '{best['name']}': {best['win_rate']:.0%} wins over {best['games']} games "
        f"({(best['win_rate'] - overall_win) * 100:+.0f} points vs your overall {overall_win:.0%}).",
        f"Your worst style is '{worst['name']}': {worst['win_rate']:.0%} wins over {worst['games']} games "
        f"({(worst['win_rate'] - overall_win) * 100:+.0f} points), and {worst['lost_on_time']:.0%} of those "
        f"games are lost on time.",
    ]
    timeouts = prof.sort_values("lost_on_time").iloc[-1]
    if timeouts.name != worst.name:
        lines.append(f"Time losses are most common in '{timeouts['name']}' games "
                     f"({timeouts['lost_on_time']:.0%}).")
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--moves", type=Path, default=LABELLED_MOVES_PATH)
    p.add_argument("--games", type=Path, default=GAMES_CSV_PATH)
    p.add_argument("--features-out", type=Path, default=GAME_FEATURES_PATH)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR / "clustering")
    args = p.parse_args(argv)

    df = build_game_features(pd.read_csv(args.moves), pd.read_csv(args.games))
    df = df.dropna(subset=STYLE_FEATURES).reset_index(drop=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[STYLE_FEATURES])

    best_k, sil = choose_k(X)
    k = args.k or best_k
    km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(X)
    df["cluster"] = km.labels_
    centres_z = pd.DataFrame(km.cluster_centers_, columns=STYLE_FEATURES)
    names = name_clusters(centres_z)
    df["cluster_name"] = df["cluster"].map(names)

    prof = profile(df, names)
    overall_win = (df["result"] == "win").mean()
    out = [
        f"=== Play-style clustering ({len(df)} games) ===",
        "Silhouette by k (higher = better separated):",
        sil.round(3).to_string(),
        f"Chosen k: {k}" + ("" if args.k is None else f" (forced; best by silhouette was {best_k})"),
        "",
        "Cluster outcomes (descriptive only, not used for clustering):",
        prof[["name", "games", "win_rate", "loss_rate", "lost_on_time", "blunder_rate",
              "mean_elo_diff"]].round(3).to_string(),
        "",
        "Cluster style profiles (means):",
        prof[STYLE_FEATURES].round(2).to_string(),
        "",
        "Distinctiveness (cluster centre in standard deviations from your average game):",
        centres_z.round(2).to_string(),
        "",
        "Plain-language findings:",
        *(f"- {s}" for s in findings(prof, overall_win)),
    ]

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.features_out, index=False)
    args.reports.mkdir(parents=True, exist_ok=True)
    sil.to_csv(args.reports / "silhouette.csv")
    prof.to_csv(args.reports / "profiles.csv")
    text = "\n".join(out)
    (args.reports / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved reports to: {args.reports}")


if __name__ == "__main__":
    main()
