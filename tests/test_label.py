"""Offline tests for blunder labelling, using a fake evaluator instead of Stockfish."""

import chess
import pandas as pd

from tilt.label import EVAL_CAP, evaluate_positions, label_moves, position_key, terminal_score
from tilt.parse import parse_game
from tests.test_parse import record

PIECES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def fake_eval(board: chess.Board) -> float:
    """White-POV: +/-cap if the side to move has mate in one, otherwise material."""
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return EVAL_CAP if board.turn == chess.WHITE else -EVAL_CAP
    return sum(v * (len(board.pieces(p, True)) - len(board.pieces(p, False))) for p, v in PIECES.items())


def scholars_mate_moves():
    _, moves = parse_game(record(), "Mr_Allez")  # 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7#
    return pd.DataFrame(moves)


def evals_for(moves, tmp_path):
    from tilt.label import positions_per_game
    return evaluate_positions(positions_per_game(moves)["fen"].tolist(), fake_eval, tmp_path / "e.csv")


def test_blunder_found_from_movers_view(tmp_path):
    moves = scholars_mate_moves()
    out = label_moves(moves, evals_for(moves, tmp_path)).set_index("san")
    assert out.loc["Nf6", "eval_before"] == 0 and out.loc["Nf6", "eval_after"] == -EVAL_CAP
    assert out.loc["Nf6", "is_blunder"] == 1
    assert out.loc["Qxf7#", "eval_drop"] == 0 and out.loc["Qxf7#", "is_blunder"] == 0
    assert out["is_blunder"].sum() == 1
    assert len(out) == len(moves)


def test_cache_is_reused(tmp_path):
    moves = scholars_mate_moves()
    evals_for(moves, tmp_path)
    calls = []
    evaluate_positions(
        moves["fen_before"].tolist(), lambda b: calls.append(b) or 0.0, tmp_path / "e.csv"
    )
    assert calls == []


def test_terminal_and_key():
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert terminal_score(mated) == -EVAL_CAP
    assert position_key("8/8/8/8/8/8/8/K6k w - - 12 80") == "8/8/8/8/8/8/8/K6k w - -"


def test_win_chance_ignores_decided_positions():
    from tilt.label import WIN_CHANCE_THRESHOLD, winning_chances
    won = winning_chances(pd.Series([9.0])) - winning_chances(pd.Series([6.0]))
    level = winning_chances(pd.Series([0.5])) - winning_chances(pd.Series([-1.5]))
    assert won[0] < WIN_CHANCE_THRESHOLD <= level[0]  # +9 -> +6 is not a blunder, +0.5 -> -1.5 is
