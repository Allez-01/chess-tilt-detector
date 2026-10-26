# Bullet Chess Tilt Detector

A **post-game** coach for my own Chess.com bullet games. It downloads my games, labels every
move with Stockfish, and uses machine learning at three levels to answer three questions:

| Level | Question | Method |
|---|---|---|
| Move | When am I most likely to blunder? | XGBoost / random forest / logistic regression |
| Game | Which kinds of games do I lose? | k-means play-style clustering |
| Session | Do I tilt, and when should I stop? | stop-rule back-tests, logistic / XGBoost / LSTM |

> **Fair play:** nothing here runs during a live game or gives real-time help. It only reads
> finished games from the official Chess.com Published-Data API, and only my own.

## Headline results (1,000 games, 1+0, Aug-Oct 2026)

- **Blunder prediction:** XGBoost reaches **ROC-AUC 0.833** on my newest games (goal 0.75) and
  catches 75% of blunders.
- **The clock is not the main problem:** with the position held fixed, a low clock raises
  blunder risk only 1.1-1.6x. Middlegames are where I blunder most.
- **Time losses are:** about 4 in 5 losses are on time. Games where my first 10 moves take
  under 15s win **54%**; slower starts win **37%**.
- **No tilt:** after 2-3 straight losses I still win about 51%; no stop rule helps, and session
  history predicts the next game no better than the rating gap.
- **Cold starts:** the first game of a session wins only **38%**.

Full write-up with every table: **[RESULTS.md](RESULTS.md)**.

## How it works

```
Chess.com API -> download.py -> parse.py -> label.py (Stockfish) -> moves_labelled.csv
                                                     |-> blunder_model.py  (move level)
                                                     |-> cluster.py        (game level)
                                                     |-> tilt_model.py     (session level)
                                                     '-> weekly_report.py + app/dashboard.py
```

Design rules: time-based splits only (train on older games, test on newer); outcome fields
(result, termination, accuracies, after-game ratings, eval drops) are blocked from features by
code and tests; class weights for rare blunders, judged by recall / F1 / ROC-AUC; `random_state=42`.

## Quickstart

```bash
conda create -n tilt python=3.11 -y
conda activate tilt
pip install -r requirements.txt
python -m pytest -q
python -m tilt.refresh --stockfish "C:\path\to\stockfish.exe" --contact you@example.com
python -m streamlit run app/dashboard.py
```

Personal game data, models and reports stay local (`data/`, `models/`, `reports/` are git-ignored).

## Pipeline details

### Step 1: download games

Run from this folder:

```bash
python -m tilt.download --username Mr_Allez --n-games 1000 --contact you@example.com
```

`--contact` (or the `CHESSCOM_CONTACT` env var) goes into the User-Agent header so
Chess.com can identify the script's owner. It is never stored in the data.

The downloader:
- makes one request at a time with a 1-second gap, never in parallel
- pauses and retries on HTTP 429 (honouring `Retry-After`) and on 5xx errors
- caches each month in `data/raw/archives/YYYY-MM.json`; finished months are never
  downloaded again, the current month is refreshed each run
- keeps standard-chess games with `time_class == "bullet"` (variants are dropped)
- walks months newest to oldest and stops once it has enough games

Output: `data/interim/bullet_games.jsonl`, one raw Chess.com game record per line,
oldest first. `data/` is git-ignored because it holds personal game data.

### Step 2: build the game and move tables

```bash
python -m tilt.parse
```

Reads `data/interim/bullet_games.jsonl` and writes two CSVs to `data/processed/`:

**games.csv** (one row per game): `game_id`, `utc_datetime` (start), `time_control`,
`colour`, pre-game `my_elo`, `opp_elo`, `elo_diff`, `eco`, `opening`, `opening_family`, `n_plies`,
plus outcome columns `result`, `my_result_code`, `termination`, `my_accuracy`. The outcome
columns are for game/session targets and analysis only.

Ratings: Chess.com's JSON `rating` is the rating *after* the game, so it reveals the result
(kept only as `my_rating_after` / `opp_rating_after`). `my_elo` is my rating after my previous
game; `opp_elo` is the opponent's after-game rating with my rating change mirrored back.

**moves.csv** (one row per ply, both sides; filter `is_me` for my moves):

| column | meaning |
|---|---|
| `ply`, `move_number` | half-move index, full-move number |
| `is_me`, `colour` | whose move it is |
| `san`, `uci`, `fen_before` | the move and the position before it (FEN feeds Stockfish in step 3) |
| `clock_before` | mover's seconds left before the move (start = base time) |
| `time_spent` | `clock_before - clock_after + increment`, floored at 0 |
| `opp_clock`, `clock_ratio` | opponent's clock at that moment, and mine / theirs |
| `material_balance` | mover's material minus opponent's, in pawns (P1 N3 B3 R5 Q9) |
| `legal_moves` | number of legal moves (complexity proxy) |
| `in_check`, `is_capture` | position / move flags |
| `phase` | `endgame` if non-pawn material (both sides) <= 26; `opening` if move <= 10 and non-pawn material >= 56; else `middlegame` |

moves.csv has no result, termination or accuracy columns, so outcome information
cannot leak into the blunder model by accident.

### Step 3: label blunders with Stockfish

Download Stockfish for Windows from https://stockfishchess.org/download/ , unzip it, and
note the path to the `.exe`. Analysis runs only on finished games, after the fact.

```bash
python -m tilt.label --stockfish "C:\path\to\stockfish.exe" --limit-games 20   # quick trial
python -m tilt.label --stockfish "C:\path\to\stockfish.exe"                     # all games
```

- Each unique position is evaluated once at depth 12 (`--depth`) and cached in
  `data/interim/evals_d12.csv`. If the run is stopped, re-running resumes where it left off.
- Scores are in pawns from the mover's point of view, capped at +/-10; a forced mate counts as 10.
- `eval_drop = eval_before - eval_after`; `is_blunder = 1` when `eval_drop >= 2` (handoff definition).
- `is_blunder_wc` (alternative): evals are turned into winning chances in [-1, 1] with the
  Lichess formula `2 / (1 + exp(-0.00368208 * centipawns)) - 1`, and a move is a blunder when
  they fall by >= 0.3. This ignores big swings in positions that stay won or lost.
- Output: `data/processed/moves_labelled.csv` = moves.csv + `eval_before`, `eval_after`,
  `eval_drop`, `wc_drop`, `is_blunder`, `is_blunder_wc`. `eval_before` is a model feature;
  `eval_after`, `eval_drop` and `wc_drop` are label-building only and must never be features.

### Step 4a: blunder model

```bash
python -m tilt.blunder_model
```

- Features (`tilt/features.py`): clock (`clock_before`, `time_spent`, `opp_clock`, `clock_ratio`),
  position (`eval_before`, `abs_eval_before`, `material_balance`, `legal_moves`, `in_check`,
  `is_capture`, `phase`), game context (`move_number`, `my_elo`, `elo_diff`, `is_white`).
  Outcome and label-building columns are blocked by an assertion.
- Target: `is_blunder_wc`. My moves only.
- Split by game start time: newest 20% of games = test; newest 20% of the rest = validation
  (used only to choose each model's F1-best threshold).
- Models: logistic regression, random forest, XGBoost, all class-weighted, `random_state=42`.
- Reports in `reports/blunder_model/`: test metrics, permutation importance, the clock effect
  with other factors held fixed (partial dependence, relative to 30s+), and plain-language findings.

### Step 4b: play-style clustering

```bash
python -m tilt.cluster          # k chosen in 3..5 by silhouette
python -m tilt.cluster --k 4    # or force k
```

- One row per game (`tilt/game_features.py`), style features only: `n_my_moves`,
  `mean_time_spent`, `fast_move_share` (<0.5s), `opening_time_used` (first 10 moves),
  `mean_clock_ratio`, `endgame_share`, `capture_share`, `mean_legal_moves`.
- Standardised, then k-means (`n_init=10`, `random_state=42`).
- Clusters are named automatically from their two most distinctive features.
- Win rate, loss rate, share lost on time and blunder rate are attached afterwards to
  describe each cluster; they are never clustering inputs.
- Writes `data/processed/game_features.csv` (reused by the tilt model) and `reports/clustering/`.

### Step 4c: sessions and tilt

```bash
python -m tilt.tilt_model            # includes the LSTM (needs TensorFlow)
python -m tilt.tilt_model --no-lstm
```

- Sessions: games starting < 30 min after the previous one ended (`tilt/sessions.py`).
- Tilt features use only EARLIER games in the session: `loss_streak`, `win_streak`, wins/losses
  so far, previous game's result / time loss / blunder rate / speed, 3-game rolling averages,
  minutes into the session, rating change in the session, plus the pre-game `elo_diff` and colour.
  A test checks that changing a game's result never changes its own features.
- Patterns: win rate, time losses, blunder rate and speed by losing streak and by game number.
- Stop rules ("stop after N straight losses", "after N games", "when K losses down") are chosen
  on the older 80% of sessions and scored on the newest 20%. Goal: kept games +5 points win rate.
  Caveat: assumes the games you'd still play would have gone the same way.
- Next-game models: rating-gap-only logistic baseline vs logistic and XGBoost on tilt features,
  and an LSTM over the previous 5 games of the session. Judged by ROC-AUC and Brier score.

### Step 5: weekly report, refresh and dashboard

```bash
python -m tilt.weekly_report        # reports/weekly/bullet_weaknesses_<date>.md
python -m tilt.refresh --stockfish "C:\path\to\stockfish.exe" --contact you@example.com
python -m streamlit run app/dashboard.py
```

- **Weekly report**: the last 7 days vs all earlier games: win rate, time losses, blunder rate
  (overall and under 10s), opening speed, slow-opening share, first-game-of-session win rate.
  Anything 10%+ worse is listed as a weakness with a concrete tip. History patterns at the bottom
  are recomputed from the data every time, never hard-coded.
- **Refresh** runs download -> parse -> label -> blunder model -> clustering -> tilt model ->
  weekly report. Cached months and evaluations make it quick after the first run.
- **Dashboard** (Streamlit): Overview, Blunders, Play-styles, Sessions and Weekly report tabs.
  It only reads saved files; it never contacts Chess.com or looks at a live game.

## Tests

```bash
python -m pytest -q
```

The tests use a fake API client and a fake evaluator, so they need neither the network nor Stockfish.

## Project layout

```
tilt/
  config.py     paths and constants (random_state = 42)
  api.py        polite Chess.com API client
  download.py   step 1 CLI
  parse.py      step 2 CLI
  label.py      step 3 CLI
  features.py   move-level features + time split
  blunder_model.py  step 4a CLI
  game_features.py  per-game summaries
  cluster.py    step 4b CLI
  sessions.py   sessions, tilt features, stop rules
  tilt_model.py step 4c CLI
  insights.py   shared analysis for report + dashboard
  weekly_report.py  step 5 weekly report
  refresh.py    run the whole pipeline
app/
  dashboard.py  Streamlit dashboard
tests/
```
