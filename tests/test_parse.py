"""Offline tests for the PGN parser, using small hand-written Chess.com-style records."""

import chess

from tilt.parse import game_phase, opening_family, opening_name, parse_game, parse_time_control

PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[White "Mr_Allez"]
[Black "Opponent"]
[Result "1-0"]
[ECO "C20"]
[ECOUrl "https://www.chess.com/openings/Kings-Pawn-Opening-Wayward-Queen-Attack-2...Nc6"]
[UTCDate "2026.10.01"]
[UTCTime "18:30:00"]
[TimeControl "{tc}"]
[Termination "Mr_Allez won by checkmate"]

1. e4 {{[%clk 0:00:59.5]}} 1... e5 {{[%clk 0:00:59.8]}} 2. Qh5 {{[%clk 0:00:58.0]}} 2... Nc6 {{[%clk 0:00:57.0]}}
3. Bc4 {{[%clk 0:00:56.5]}} 3... Nf6 {{[%clk 0:00:55.0]}} 4. Qxf7# {{[%clk 0:00:55.9]}} 1-0
"""


def record(tc="60", me_white=True):
    me = {"username": "Mr_Allez", "rating": 1500, "result": "win"}
    opp = {"username": "Opponent", "rating": 1450, "result": "checkmated"}
    white, black = (me, opp) if me_white else (opp, me)
    return {
        "url": "https://www.chess.com/game/live/123456",
        "pgn": PGN.format(tc=tc),
        "time_control": tc,
        "end_time": 1790000000,
        "white": white,
        "black": black,
    }


def test_game_row():
    game, moves = parse_game(record(), "mr_allez")
    assert game["game_id"] == "123456"
    assert game["colour"] == "white" and game["result"] == "win"
    assert game["my_rating_after"] == 1500 and "elo_diff" not in game
    assert game["utc_datetime"].startswith("2026-10-01T18:30:00")
    assert game["opening_family"] == "Kings Pawn Opening"
    assert game["n_plies"] == 7 and game["n_my_moves"] == 4


def test_clock_features_no_increment():
    _, moves = parse_game(record(), "Mr_Allez")
    first, second_mine = moves[0], moves[2]
    assert first["clock_before"] == 60 and first["time_spent"] == 0.5
    assert second_mine["clock_before"] == 59.5 and second_mine["time_spent"] == 1.5
    assert second_mine["opp_clock"] == 59.8
    assert abs(second_mine["clock_ratio"] - round(59.5 / 59.8, 3)) < 1e-9


def test_increment_added_back():
    _, moves = parse_game(record(tc="120+1"), "Mr_Allez")
    # last move: 56.5 -> 55.9 with +1 increment means 1.6s actually spent
    assert moves[-1]["time_spent"] == 1.6


def test_position_features():
    _, moves = parse_game(record(), "Mr_Allez")
    mate = moves[-1]
    assert mate["san"] == "Qxf7#" and mate["is_capture"] and mate["is_me"]
    assert mate["material_balance"] == 0 and mate["phase"] == "opening"
    assert moves[1]["is_me"] is False


def test_no_outcome_columns_in_moves():
    _, moves = parse_game(record(), "Mr_Allez")
    leaky = {"result", "termination", "my_accuracy", "eval_drop"}
    assert not leaky & set(moves[0])


def test_helpers():
    assert parse_time_control("60") == (60, 0)
    assert parse_time_control("120+1") == (120, 1)
    assert opening_name("https://www.chess.com/openings/Sicilian-Defense-2...d6") == "Sicilian Defense"
    assert opening_family("Sicilian Defense Najdorf Variation") == "Sicilian Defense"
    assert game_phase(chess.Board("8/5k2/8/8/8/2R5/5K2/8 w - - 0 40")) == "endgame"


def test_pregame_ratings_never_use_this_games_result():
    import pandas as pd
    from tilt.parse import add_pregame_ratings
    games = pd.DataFrame({
        "utc_datetime": ["2026-10-01T10:00", "2026-10-01T10:02", "2026-10-01T10:04"],
        "my_elo_header": [1500, 1508, 1500], "my_rating_after": [1508, 1500, 1507],
        "opp_rating_after": [1492, 1610, 1400],
    })
    g = add_pregame_ratings(games)
    assert g["my_elo"].tolist() == [1500, 1508, 1500]   # previous game's after-rating
    assert g["opp_elo"].tolist() == [1500, 1602, 1407]  # undo the mirrored rating change
    assert g["elo_diff"].tolist() == [0, -94, 93]
