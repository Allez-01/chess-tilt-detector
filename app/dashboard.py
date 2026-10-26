"""Bullet Tilt Detector dashboard (post-game only).

Run from the project root:
    python -m streamlit run app/dashboard.py

Reads the files written by steps 2-5. It never connects to Chess.com and never looks at a
game in progress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tilt.config import (  # noqa: E402
    LABELLED_MOVES_PATH, REPORTS_DIR, SESSION_GAMES_PATH,
)
from tilt.insights import (  # noqa: E402
    blunder_rate_by_clock, blunder_rate_by_phase, load_games, load_my_moves,
    standing_findings, win_rate_by_game_number, win_rate_by_loss_streak,
)
from tilt.weekly_report import build_report  # noqa: E402

st.set_page_config(page_title="Bullet Tilt Detector", page_icon="♞", layout="wide")

# Single-series charts use one categorical slot (reference palette, blue), stepped per theme.
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


def bar(df: pd.DataFrame, x: str, y: str, y_title: str, pct: bool = True,
        horizontal: bool = False, sort=None) -> alt.Chart:
    fmt = ".0%" if pct else ".2f"
    cat = alt.X(f"{x}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0))
    val = alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=fmt, grid=True, gridOpacity=0.3))
    if horizontal:
        cat = alt.Y(f"{x}:N", sort="-x", title=None)
        val = alt.X(f"{y}:Q", title=y_title, axis=alt.Axis(format=fmt, grid=True, gridOpacity=0.3))
    radius = ({"cornerRadiusTopRight": 4, "cornerRadiusBottomRight": 4} if horizontal
              else {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4})
    tooltip = [alt.Tooltip(f"{c}:Q", format=fmt if c == y else ",") if pd.api.types.is_numeric_dtype(df[c])
               else alt.Tooltip(f"{c}:N") for c in df.columns]
    height = max(280, 28 * len(df)) if horizontal else 280
    return (alt.Chart(df).mark_bar(color=SERIES[theme()], size=16 if horizontal else 28, **radius)
            .encode(x=cat if not horizontal else val, y=val if not horizontal else cat, tooltip=tooltip)
            .properties(height=height))


def need(path: Path, command: str) -> bool:
    if path.exists():
        return True
    st.info(f"`{path.name}` not found. Run `{command}` first.")
    return False


@st.cache_data
def data():
    return load_games(), load_my_moves()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("♞ Bullet Tilt Detector")
st.sidebar.caption("Post-game coach for my own Chess.com bullet games. "
                   "Reads finished games only; nothing here runs during a live game.")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()

if not (need(SESSION_GAMES_PATH, "python -m tilt.refresh")
        and need(LABELLED_MOVES_PATH, "python -m tilt.label")):
    st.stop()
games, my_moves = data()
st.sidebar.write(f"{len(games)} games, {games['start'].min():%d %b} to {games['start'].max():%d %b %Y}")

overview, blunders, styles, sessions, weekly = st.tabs(
    ["Overview", "Blunders", "Play-styles", "Sessions", "Weekly report"])

# ---------------------------------------------------------------- overview
with overview:
    st.header("Overview")
    c = st.columns(4)
    c[0].metric("Games", f"{len(games):,}")
    c[1].metric("Win rate", f"{games['won'].mean():.0%}")
    c[2].metric("Games lost on time", f"{games['lost_on_time'].mean():.0%}")
    c[3].metric("Blunder rate (my moves)", f"{my_moves['is_blunder_wc'].mean():.1%}")

    st.subheader("What your games say")
    st.markdown("\n".join(f"- {line}" for line in standing_findings(games)))

# ---------------------------------------------------------------- blunders
with blunders:
    st.header("Blunders")
    left, right = st.columns(2)
    with left:
        st.subheader("Blunder rate by clock left")
        clock = blunder_rate_by_clock(my_moves)
        st.altair_chart(bar(clock, "clock", "blunder_rate", "Blunder rate", sort=list(clock["clock"])),
                        width="stretch")
    with right:
        st.subheader("Blunder rate by phase")
        phase = blunder_rate_by_phase(my_moves)
        st.altair_chart(bar(phase, "phase", "blunder_rate", "Blunder rate", sort=list(phase["phase"])),
                        width="stretch")

    rep = REPORTS_DIR / "blunder_model"
    if need(rep / "clock_effect.csv", "python -m tilt.blunder_model"):
        left, right = st.columns(2)
        with left:
            st.subheader("Clock effect, other factors held fixed")
            st.caption("Model-estimated blunder risk relative to having 30s+ left (1.0 = same risk).")
            eff = pd.read_csv(rep / "clock_effect.csv")[["bucket", "all_vs_30s"]]
            st.altair_chart(bar(eff, "bucket", "all_vs_30s", "Relative risk", pct=False,
                                sort=list(eff["bucket"])), width="stretch")
        with right:
            st.subheader("What the model leans on")
            st.caption("Drop in test ROC-AUC when each feature is shuffled.")
            imp = pd.read_csv(rep / "feature_importance.csv").head(10)[["feature", "auc_drop"]]
            st.altair_chart(bar(imp, "feature", "auc_drop", "ROC-AUC drop", pct=False, horizontal=True),
                            width="stretch")
        st.subheader("Model test metrics (newest 20% of games)")
        st.dataframe(pd.read_csv(rep / "metrics.csv", index_col=0).round(3))

# ---------------------------------------------------------------- play-styles
with styles:
    st.header("Play-styles")
    path = REPORTS_DIR / "clustering" / "profiles.csv"
    if need(path, "python -m tilt.cluster"):
        prof = pd.read_csv(path, index_col=0)
        st.subheader("Win rate by style")
        st.altair_chart(bar(prof[["name", "win_rate", "games", "lost_on_time"]], "name", "win_rate",
                            "Win rate", sort="-y"), width="stretch")
        st.subheader("Style profiles")
        st.dataframe(prof.round(3))

# ---------------------------------------------------------------- sessions
with sessions:
    st.header("Sessions and tilt")
    left, right = st.columns(2)
    with left:
        st.subheader("Win rate by game number in session")
        gn = win_rate_by_game_number(games)
        st.altair_chart(bar(gn, "game in session", "win_rate", "Win rate",
                            sort=list(gn["game in session"])), width="stretch")
    with right:
        st.subheader("Win rate by losses in a row")
        ls = win_rate_by_loss_streak(games)
        st.altair_chart(bar(ls, "losses in a row", "win_rate", "Win rate",
                            sort=list(ls["losses in a row"])), width="stretch")
    path = REPORTS_DIR / "tilt" / "rules.csv"
    if need(path, "python -m tilt.tilt_model"):
        st.subheader("Stop rules back-test (all games)")
        st.dataframe(pd.read_csv(path, index_col=0)[
            ["kept_games", "kept_share", "kept_win_rate", "dropped_win_rate", "gain_points"]].round(3))

# ---------------------------------------------------------------- weekly report
with weekly:
    st.header("Weekly bullet weaknesses")
    out_dir = REPORTS_DIR / "weekly"
    if st.button("Build report for the latest week"):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"bullet_weaknesses_{games['start'].max():%Y-%m-%d}.md").write_text(
            build_report(games, my_moves), encoding="utf-8")
    files = sorted(out_dir.glob("bullet_weaknesses_*.md"), reverse=True) if out_dir.exists() else []
    if files:
        choice = st.selectbox("Week", files, format_func=lambda p: p.stem.replace("bullet_weaknesses_", ""))
        st.markdown(choice.read_text(encoding="utf-8"))
    else:
        st.info("No weekly report yet. Click the button above or run `python -m tilt.weekly_report`.")
