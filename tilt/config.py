"""Shared paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_CACHE_DIR = DATA_DIR / "raw" / "archives"
BULLET_GAMES_PATH = DATA_DIR / "interim" / "bullet_games.jsonl"

DEFAULT_USERNAME = "Mr_Allez"
DEFAULT_N_GAMES = 1000
RANDOM_STATE = 42

PROCESSED_DIR = DATA_DIR / "processed"
GAMES_CSV_PATH = PROCESSED_DIR / "games.csv"
MOVES_CSV_PATH = PROCESSED_DIR / "moves.csv"
LABELLED_MOVES_PATH = PROCESSED_DIR / "moves_labelled.csv"
EVAL_CACHE_DIR = DATA_DIR / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
GAME_FEATURES_PATH = PROCESSED_DIR / "game_features.csv"
SESSION_GAMES_PATH = PROCESSED_DIR / "session_games.csv"
