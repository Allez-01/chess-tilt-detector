# Results: what 1,000 of my bullet games say

Data: my 1,000 most recent Chess.com bullet games (all 1+0), 28 Aug to 23 Oct 2026,
downloaded through the official Published-Data API. 65,037 moves in total, 32,519 of them mine.
Win / loss / draw: 498 / 482 / 20. Every number below comes from the pipeline's own output.

## Scorecard against the project goals

| Goal | Target | Result | Status |
|---|---|---|---|
| 1. Predict blunders | ROC-AUC >= 0.75 on newest 20% of games | XGBoost **0.833** (RF 0.828, LR 0.774) | Met |
| 2. Clock danger zone | a clock bucket where blunder risk at least doubles | with other factors held fixed, risk rises only **1.1-1.6x** below 30s | Not met |
| 3. Play-styles | 3-5 interpretable clusters with win rates | **3 styles**, win rates **38% to 73%** | Met |
| 4. Stop rule | kept games win >= 5 points more | best rule **-0.9 points** on newest sessions | Not met (no tilt found) |
| 5. Fair and private | finished games only, own data only | post-game only; personal data git-ignored | Met |

## 1. Blunders (move level)

**Label.** Stockfish (depth 12) evaluates every position. A move is a blunder when it drops
my winning chances by at least 0.3 on a -1..1 scale (the Lichess method). The handoff's
original "lost 2+ pawns" rule was compared first: it flagged 11.7% of my moves, and half of
its flags in already-decided positions (|eval| >= 5) never changed the result (10.5% vs 5.2%).
The win-chance label flags 9.4% of my moves.

**Model.** Time-based split by game (train on older games, test on the newest 200). Class
weights handle the imbalance; thresholds are tuned on a validation slice, never on test.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic regression | 0.774 | 0.242 | 0.542 | 0.224 | 0.317 |
| Random forest | 0.828 | 0.297 | 0.493 | 0.283 | 0.360 |
| **XGBoost** | **0.833** | **0.327** | **0.746** | 0.228 | 0.349 |

Base blunder rate on the test games: 9.2%, so XGBoost's precision is about 2.5x random.

**What drives it** (permutation importance, drop in test ROC-AUC): `eval_before` 0.193,
`material_balance` 0.075, `is_capture` 0.027, `clock_before` 0.026. The position matters far
more than the clock. Part of `eval_before`'s weight is mechanical: you can only lose a lot of
winning chance from a position that isn't already lost.

**Clock effect, other factors held fixed** (partial dependence, risk vs 30s+ left):

| Clock left | All | Opening | Middlegame | Endgame |
|---|---|---|---|---|
| <3s | 1.41x | 1.48x | 1.32x | 1.64x |
| 3-5s | 1.30x | 1.31x | 1.24x | 1.51x |
| 5-10s | 1.26x | 1.30x | 1.24x | 1.28x |
| 10-30s | 1.17x | 1.25x | 1.14x | 1.15x |

**By phase** (raw win-chance blunder rate): opening 5.8%, middlegame 12.9%, endgame 6.9%.

## 2. Play-styles (game level)

k-means on eight style features per game (speed, premove share, opening time, clock lead,
endgame share, captures, complexity, length); k = 3 chosen by silhouette (0.18, so these are
broad tendencies, not sharp types). Results were attached only after clustering.

| Style | Games | Win rate | Lost on time | First 10 moves | Clock ratio |
|---|---|---|---|---|---|
| Fast opening, few endgames | 303 | **73%** | 15% | 8.8s | 1.27 |
| Endgame-heavy, long games | 335 | 41% | 48% | 10.7s | 0.99 |
| Slow moves, slow opening | 362 | **38%** | 48% | 16.6s | 0.92 |

About 380 of my 482 losses are on time. Average rating gaps per style range only from -2 to +8,
so opponent strength doesn't explain the difference.

## 3. Sessions and tilt (session level)

76 sessions (gaps under 30 minutes), median 11 games. Every feature uses only earlier games in
the session; a unit test checks that a game's own result can never change its features.

| Losses in a row before the game | Games | Win rate |
|---|---|---|
| 0 | 563 | 48.0% |
| 1 | 262 | 53.1% |
| 2 | 104 | 51.0% |
| 3+ | 71 | 50.7% |

| Game in session | Games | Win rate | Lost on time |
|---|---|---|---|
| 1st | 76 | **38.2%** | 52.6% |
| 2-5 | 271 | 50.6% | 41.0% |
| 6-10 | 265 | 50.6% | 38.1% |
| 11-20 | 270 | 49.6% | 34.8% |
| 21+ | 118 | 54.2% | 28.8% |

**Stop rules.** Twelve rules were back-tested; none beat the 49.8% baseline by more than
0.3 points. The rule chosen on the older 61 sessions ("stop after 3 straight losses") scored
-0.9 points on the newest 15.

**Next-game prediction** (newest 15 sessions, 209 games):

| Model | ROC-AUC |
|---|---|
| Rating gap only (baseline) | 0.584 |
| Logistic regression, session features | 0.519 |
| XGBoost, session features | 0.511 |
| LSTM over the previous 5 games | 0.509 |

Session history adds nothing beyond the rating gap: I don't tilt.

## 4. Coaching takeaways

1. **Play the first 10 moves fast.** Under 15s: 54% wins. Slower: 37%.
2. **Warm up before rated games.** First game of a session: 38% wins and over half lost on time.
3. **Don't quit after losses.** Losing streaks don't lower my next-game win rate.
4. **Respect the middlegame.** It has more than double the blunder rate of the opening.

## Lessons from building it

- **A leak hidden in the API.** Chess.com's game `rating` (and the PGN `WhiteElo`/`BlackElo`
  headers) are the ratings *after* the game. Using them as "pre-game" ratings let a
  rating-gap-only model predict results at ROC-AUC 0.943. Rebuilding pre-game ratings from
  the previous game dropped it to an honest 0.584. A warning now fires if it ever looks too good.
- **Label choice matters.** The simple "2 pawns" blunder rule over-counted harmless moves in
  won or lost positions; a win-chance label fixed that.
- **Negative results are results.** No tilt and no doubling clock threshold were found, and the
  write-up says so rather than tuning until something looks good.
