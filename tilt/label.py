"""Step 3: label blunders with a local Stockfish engine (finished games only).

Usage (from the project root):
    python -m tilt.label --stockfish C:\\path\\to\\stockfish.exe --depth 12

Reads:
    data/processed/moves.csv            (from step 2)
Writes:
    data/interim/evals_d{depth}.csv     cache of position evaluations; a stopped run resumes from here
    data/processed/moves_labelled.csv   moves.csv + eval_before, eval_after, eval_drop, is_blunder

All evaluations are in pawns from the MOVER's point of view, capped at +/- EVAL_CAP
(a forced mate counts as the cap). A move is a blunder when
    eval_drop = eval_before - eval_after >= 2 pawns.

eval_drop builds the label only. It must never be used as a model feature.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import chess
import chess.engine
import numpy as np
import pandas as pd

from tilt.config import EVAL_CACHE_DIR, LABELLED_MOVES_PATH, MOVES_CSV_PATH

log = logging.getLogger(__name__)

EVAL_CAP = 10.0          # pawns; also the value of a forced mate
BLUNDER_THRESHOLD = 2.0  # pawns of eval lost by the move
# Win-chance label (Lichess method): winning chances in [-1, 1] from centipawns,
# blunder when they fall by >= 0.3 (= 15 percentage points of win probability).
WIN_CHANCE_SLOPE = 0.00368208
WIN_CHANCE_THRESHOLD = 0.3

# An evaluator takes a position and returns its score in pawns from WHITE's point of view.
Evaluator = Callable[[chess.Board], float]


def position_key(fen: str) -> str:
    """Board, side to move, castling and en passant. Move counters don't change the eval."""
    return " ".join(fen.split()[:4])


def winning_chances(pawns: pd.Series) -> pd.Series:
    return 2 / (1 + np.exp(-WIN_CHANCE_SLOPE * pawns * 100)) - 1


def cap(x: float) -> float:
    return max(-EVAL_CAP, min(EVAL_CAP, x))


def terminal_score(board: chess.Board) -> float | None:
    """White-POV score for finished positions, so the engine is never asked about them."""
    if board.is_checkmate():
        return -EVAL_CAP if board.turn == chess.WHITE else EVAL_CAP
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    return None


def stockfish_evaluator(engine: chess.engine.SimpleEngine, depth: int) -> Evaluator:
    def evaluate(board: chess.Board) -> float:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            return EVAL_CAP if score.mate() > 0 else -EVAL_CAP
        return cap(score.score() / 100)
    return evaluate


# ---------------------------------------------------------------- positions to evaluate

def positions_per_game(moves: pd.DataFrame) -> pd.DataFrame:
    """Every position before each ply, plus the final position after each game's last ply.

    Returns columns: game_id, ply, fen (the position BEFORE that ply; the final position
    gets ply = last_ply + 1).
    """
    before = moves[["game_id", "ply", "fen_before", "uci"]].rename(columns={"fen_before": "fen"})
    last = moves.sort_values("ply").groupby("game_id", sort=False).tail(1)
    finals = []
    for row in last.itertuples(index=False):
        board = chess.Board(row.fen_before)
        board.push_uci(row.uci)
        finals.append({"game_id": row.game_id, "ply": row.ply + 1, "fen": board.fen()})
    return pd.concat([before[["game_id", "ply", "fen"]], pd.DataFrame(finals)], ignore_index=True)


# ---------------------------------------------------------------- eval cache

def load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["key"], df["eval_white"]))


def evaluate_positions(
    fens: list[str],
    evaluate: Evaluator,
    cache_path: Path,
    flush_every: int = 200,
) -> dict[str, float]:
    """White-POV eval for every unique position, appending new results to the cache file."""
    cache = load_cache(cache_path)
    todo = [k for k in dict.fromkeys(position_key(f) for f in fens) if k not in cache]
    log.info("Unique positions: %d, already cached: %d, to evaluate: %d",
             len(cache) + len(todo), len(cache), len(todo))
    if not todo:
        return cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not cache_path.exists()
    start = time.monotonic()
    with cache_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["key", "eval_white"])
        for i, key in enumerate(todo, start=1):
            board = chess.Board(key + " 0 1")
            score = terminal_score(board)
            if score is None:
                score = evaluate(board)
            cache[key] = score
            writer.writerow([key, round(score, 2)])
            if i % flush_every == 0 or i == len(todo):
                f.flush()
                rate = i / (time.monotonic() - start)
                log.info("Evaluated %d / %d (%.0f pos/s, ~%.0f min left)",
                         i, len(todo), rate, (len(todo) - i) / rate / 60)
    return cache


# ---------------------------------------------------------------- labelling

def label_moves(moves: pd.DataFrame, evals_white: dict[str, float]) -> pd.DataFrame:
    """Add eval_before, eval_after, eval_drop (mover POV, pawns), wc_drop and both blunder labels.

    is_blunder:    eval_drop >= 2 pawns (the handoff definition)
    is_blunder_wc: winning chances drop >= 0.3 (ignores big swings in already-decided positions)
    """
    pos = positions_per_game(moves)
    pos["eval_white"] = pos["fen"].map(lambda f: evals_white[position_key(f)])
    after = pos.assign(ply=pos["ply"] - 1)[["game_id", "ply", "eval_white"]]

    out = moves.merge(
        pos[["game_id", "ply", "eval_white"]], on=["game_id", "ply"], how="left"
    ).merge(after, on=["game_id", "ply"], how="left", suffixes=("_before", "_after"))

    sign = out["colour"].map({"white": 1.0, "black": -1.0})
    out["eval_before"] = (out["eval_white_before"] * sign).round(2)
    out["eval_after"] = (out["eval_white_after"] * sign).round(2)
    out["eval_drop"] = (out["eval_before"] - out["eval_after"]).round(2)
    out["is_blunder"] = (out["eval_drop"] >= BLUNDER_THRESHOLD).astype(int)
    out["wc_drop"] = (winning_chances(out["eval_before"]) - winning_chances(out["eval_after"])).round(3)
    out["is_blunder_wc"] = (out["wc_drop"] >= WIN_CHANCE_THRESHOLD).astype(int)
    return out.drop(columns=["eval_white_before", "eval_white_after"])


LABELS = ["is_blunder", "is_blunder_wc"]


def rate_table(mine: pd.DataFrame, groups: pd.Series, title: str) -> list[str]:
    g = mine.groupby(groups, observed=False)[LABELS]
    sizes, means = g.size(), g.mean()
    lines = [f"{title:<22}{'2-pawn':>9}{'win-chance':>12}{'n':>8}"]
    for key in sizes.index:
        if sizes[key]:
            lines.append(f"  {str(key):<20}{means.loc[key, LABELS[0]]:>9.2%}"
                         f"{means.loc[key, LABELS[1]]:>12.2%}{sizes[key]:>8}")
        else:
            lines.append(f"  {str(key):<20}{'-':>9}{'-':>12}{0:>8}")
    return lines


def summarise(labelled: pd.DataFrame) -> str:
    mine = labelled[labelled["is_me"]]
    theirs = labelled[~labelled["is_me"]]
    clock = pd.cut(mine["clock_before"], [0, 3, 5, 10, 30, 61], right=False,
                   labels=["<3s", "3-5s", "5-10s", "10-30s", "30s+"])
    decided = pd.cut(mine["eval_before"].abs(), [0, 2, 5, 10.01], right=False,
                     labels=["level |eval|<2", "clear 2-5", "decided 5+"])
    lines = [
        "=== Label summary ===",
        f"Moves labelled:    {len(labelled)}  (mine: {len(mine)})",
        f"{'':<22}{'2-pawn':>9}{'win-chance':>12}",
        f"  {'my blunders':<20}{mine[LABELS[0]].sum():>9}{mine[LABELS[1]].sum():>12}",
        f"  {'my blunder rate':<20}{mine[LABELS[0]].mean():>9.2%}{mine[LABELS[1]].mean():>12.2%}",
        f"  {'opponent blunders':<20}{theirs[LABELS[0]].sum():>9}{theirs[LABELS[1]].sum():>12}",
        *rate_table(mine, clock, "By clock_before:"),
        *rate_table(mine, mine["phase"], "By phase:"),
        *rate_table(mine, decided, "By position before:"),
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"),
                   help="Path to the Stockfish executable (or set STOCKFISH_PATH).")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=128)
    p.add_argument("--moves", type=Path, default=MOVES_CSV_PATH)
    p.add_argument("--out", type=Path, default=LABELLED_MOVES_PATH)
    p.add_argument("--cache-dir", type=Path, default=EVAL_CACHE_DIR)
    p.add_argument("--limit-games", type=int, default=None,
                   help="Only label the first N games (quick trial run).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    moves = pd.read_csv(args.moves)
    if args.limit_games:
        keep = moves["game_id"].drop_duplicates().head(args.limit_games)
        moves = moves[moves["game_id"].isin(keep)]

    if not (Path(args.stockfish).is_file() or shutil.which(args.stockfish)):
        raise SystemExit(
            f"Stockfish not found at: {args.stockfish}\n"
            "Pass the full path to the .exe, e.g. --stockfish \"C:\\...\\stockfish-windows-x86-64-avx2.exe\" "
            "(in File Explorer: right-click the file -> Copy as path)."
        )
    cache_path = args.cache_dir / f"evals_d{args.depth}.csv"
    fens = positions_per_game(moves)["fen"].tolist()
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        evals = evaluate_positions(fens, stockfish_evaluator(engine, args.depth), cache_path)

    labelled = label_moves(moves, evals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(args.out, index=False)
    print(summarise(labelled))
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
