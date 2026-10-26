"""Step 2: parse downloaded PGNs into a game table and a move-level table.

Usage (from the project root):
    python -m tilt.parse

Reads:
    data/interim/bullet_games.jsonl   (from step 1)
Writes:
    data/processed/games.csv   one row per game (headers, ratings, result)
    data/processed/moves.csv   one row per ply, both sides, with clock and position features

Leakage note: moves.csv deliberately has no result/termination/accuracy columns.
Those live only in games.csv and must never be joined in as move-level features.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

from tilt.config import BULLET_GAMES_PATH, DEFAULT_USERNAME, GAMES_CSV_PATH, MOVES_CSV_PATH

log = logging.getLogger(__name__)

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

# Chess.com per-player result codes -> my result
DRAW_CODES = {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"}

# Phase thresholds on non-pawn material (both sides, N/B=3, R=5, Q=9; full board = 62)
ENDGAME_NPM = 26        # e.g. each side down to a rook + minor piece, or queens off plus trades
OPENING_MAX_MOVE = 10   # opening = first 10 full moves while little material has been traded
OPENING_MIN_NPM = 56

OPENING_FAMILY_WORDS = {"Defense", "Opening", "Game", "Gambit", "Attack", "System"}


# ---------------------------------------------------------------- position features

def material(board: chess.Board, colour: chess.Color) -> int:
    return sum(v * len(board.pieces(p, colour)) for p, v in PIECE_VALUES.items())


def non_pawn_material(board: chess.Board) -> int:
    return sum(
        v * (len(board.pieces(p, chess.WHITE)) + len(board.pieces(p, chess.BLACK)))
        for p, v in PIECE_VALUES.items()
        if p != chess.PAWN
    )


def game_phase(board: chess.Board) -> str:
    npm = non_pawn_material(board)
    if npm <= ENDGAME_NPM:
        return "endgame"
    if board.fullmove_number <= OPENING_MAX_MOVE and npm >= OPENING_MIN_NPM:
        return "opening"
    return "middlegame"


# ---------------------------------------------------------------- game-level helpers

def parse_time_control(tc: str) -> tuple[int, int]:
    """'60' -> (60, 0); '120+1' -> (120, 1)."""
    base, _, inc = (tc or "0").partition("+")
    return int(base), int(inc or 0)


def my_result(code: str) -> str:
    if code == "win":
        return "win"
    if code in DRAW_CODES:
        return "draw"
    return "loss"


def opening_name(eco_url: str) -> str:
    """'https://www.chess.com/openings/Sicilian-Defense-Bowdlers-Attack' -> 'Sicilian Defense Bowdlers Attack'."""
    slug = (eco_url or "").rstrip("/").split("/")[-1]
    slug = re.split(r"-\d+\.", slug)[0]  # drop trailing move sequences like '-2...a6'
    return slug.replace("-", " ").strip()


def opening_family(name: str) -> str:
    """Coarse grouping: words up to the first 'Defense/Opening/Game/...', else the first two words."""
    words = name.split()
    for i, w in enumerate(words):
        if w in OPENING_FAMILY_WORDS:
            return " ".join(words[: i + 1])
    return " ".join(words[:2])


def start_datetime(headers: chess.pgn.Headers, fallback_ts: int) -> datetime:
    date, time_ = headers.get("UTCDate"), headers.get("UTCTime")
    if date and time_ and "?" not in date:
        return datetime.strptime(f"{date} {time_}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(fallback_ts, timezone.utc)


# ---------------------------------------------------------------- per-game parsing

def parse_game(raw: dict, username: str) -> tuple[dict, list[dict]]:
    """Return (game_row, move_rows) for one Chess.com game record."""
    pgn = chess.pgn.read_game(io.StringIO(raw["pgn"]))
    headers = pgn.headers
    me_white = raw["white"]["username"].lower() == username.lower()
    me, opp = ("white", "black") if me_white else ("black", "white")
    my_colour = chess.WHITE if me_white else chess.BLACK

    base, inc = parse_time_control(raw.get("time_control", headers.get("TimeControl", "")))
    game_id = raw["url"].rstrip("/").split("/")[-1]
    opening = opening_name(headers.get("ECOUrl", ""))

    moves: list[dict] = []
    clocks = {chess.WHITE: float(base), chess.BLACK: float(base)}  # last known clock per side
    board = pgn.board()
    for ply, node in enumerate(pgn.mainline(), start=1):
        move = node.move
        mover = board.turn
        clock_before = clocks[mover]
        opp_clock = clocks[not mover]
        clock_after = node.clock()
        if clock_after is None:  # missing %clk: carry the previous value so the row stays usable
            clock_after = clock_before
        time_spent = max(0.0, clock_before - clock_after + inc)

        moves.append(
            {
                "game_id": game_id,
                "ply": ply,
                "move_number": board.fullmove_number,
                "is_me": mover == my_colour,
                "colour": "white" if mover == chess.WHITE else "black",
                "san": board.san(move),
                "uci": move.uci(),
                "fen_before": board.fen(),
                "clock_before": clock_before,
                "time_spent": round(time_spent, 2),
                "opp_clock": opp_clock,
                "clock_ratio": round(clock_before / opp_clock, 3) if opp_clock > 0 else None,
                "material_balance": material(board, mover) - material(board, not mover),
                "legal_moves": board.legal_moves.count(),
                "in_check": board.is_check(),
                "is_capture": board.is_capture(move),
                "phase": game_phase(board),
            }
        )
        clocks[mover] = clock_after
        board.push(move)

    accuracies = raw.get("accuracies") or {}
    game = {
        "game_id": game_id,
        "url": raw["url"],
        "utc_datetime": start_datetime(headers, raw["end_time"]).isoformat(),
        "end_datetime": datetime.fromtimestamp(raw["end_time"], timezone.utc).isoformat(),
        "time_control": raw.get("time_control"),
        "base_seconds": base,
        "increment": inc,
        "colour": me,
        "my_elo_header": _int(headers.get(f"{me.capitalize()}Elo")),
        "opp_elo_header": _int(headers.get(f"{opp.capitalize()}Elo")),
        "eco": headers.get("ECO"),
        "opening": opening,
        "opening_family": opening_family(opening),
        "n_plies": len(moves),
        "n_my_moves": sum(m["is_me"] for m in moves),
        # --- outcome columns: game/session targets and analysis only, never move-level features
        "result": my_result(raw[me]["result"]),
        "my_result_code": raw[me]["result"],
        "opp_result_code": raw[opp]["result"],
        "termination": headers.get("Termination"),
        "my_accuracy": accuracies.get(me),
        "opp_accuracy": accuracies.get(opp),
        # Chess.com's JSON "rating" is the rating AFTER the game, so it reveals the result.
        "my_rating_after": raw[me].get("rating"),
        "opp_rating_after": raw[opp].get("rating"),
    }
    return game, moves


def _int(value: str | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def add_pregame_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Pre-game ratings that cannot reveal the result.

    my_elo:  my rating after my PREVIOUS bullet game (games are consecutive), or the PGN header
             for the first game in the data.
    opp_elo: opponent's after-game rating minus the points they gained, approximated as the
             mirror of my own change (rating exchanges are close to zero-sum).
    """
    g = games.sort_values("utc_datetime").reset_index(drop=True)
    g["my_elo"] = g["my_rating_after"].astype(float).shift().fillna(g["my_elo_header"].astype(float))
    my_change = (g["my_rating_after"] - g["my_elo"]).fillna(0)
    g["opp_elo"] = g["opp_rating_after"] + my_change
    g["elo_diff"] = g["my_elo"] - g["opp_elo"]
    return g


def rating_check(games: pd.DataFrame) -> str:
    """How the PGN header ratings relate to Chess.com's after-game ratings (for the record)."""
    same_as_after = (games["my_elo_header"] == games["my_rating_after"]).mean()
    same_as_before = (games["my_elo_header"] == games["my_rating_after"].shift()).iloc[1:].mean()
    return (f"Rating check:      PGN header = after-game rating in {same_as_after:.0%} of games, "
            f"= previous game's rating in {same_as_before:.0%}")


def build_tables(games_path: Path, username: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_rows, move_rows = [], []
    with games_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = json.loads(line)
            try:
                game, moves = parse_game(raw, username)
            except Exception as exc:  # one bad PGN shouldn't stop the run
                log.warning("Skipping line %d (%s): %s", line_no, raw.get("url"), exc)
                continue
            game_rows.append(game)
            move_rows.extend(moves)

    games = add_pregame_ratings(pd.DataFrame(game_rows))
    moves = pd.DataFrame(move_rows)
    return games, moves


def summarise(games: pd.DataFrame, moves: pd.DataFrame) -> str:
    mine = moves[moves["is_me"]]
    lines = [
        "=== Parse summary ===",
        f"Games parsed:      {len(games)}  (with 0 moves: {(games['n_plies'] == 0).sum()})",
        f"Move rows (plies): {len(moves)}  (mine: {len(mine)})",
        "Results:           " + ", ".join(f"{k}: {v}" for k, v in games["result"].value_counts().items()),
        "My moves by phase: " + ", ".join(f"{k}: {v}" for k, v in mine["phase"].value_counts().items()),
        "My clock_before:   " + ", ".join(
            f"{q}: {mine['clock_before'].quantile(p):.1f}s"
            for q, p in [("min", 0), ("median", 0.5), ("max", 1)]
        ),
        "My time_spent:     " + ", ".join(
            f"{q}: {mine['time_spent'].quantile(p):.2f}s"
            for q, p in [("median", 0.5), ("p95", 0.95), ("max", 1)]
        ),
        f"My moves under 5s: {(mine['clock_before'] < 5).sum()}",
        rating_check(games),
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--username", default=DEFAULT_USERNAME)
    p.add_argument("--games", type=Path, default=BULLET_GAMES_PATH)
    p.add_argument("--games-out", type=Path, default=GAMES_CSV_PATH)
    p.add_argument("--moves-out", type=Path, default=MOVES_CSV_PATH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    games, moves = build_tables(args.games, args.username)
    args.games_out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.games_out, index=False)
    moves.to_csv(args.moves_out, index=False)
    print(summarise(games, moves))
    print(f"Saved to: {args.games_out}\n          {args.moves_out}")


if __name__ == "__main__":
    main()
