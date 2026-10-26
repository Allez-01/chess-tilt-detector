"""Run the whole post-game pipeline in order, e.g. once a week after playing.

Usage (from the project root):
    python -m tilt.refresh --stockfish "C:\\...\\stockfish.exe" --contact you@example.com

Steps: download -> parse -> label -> blunder model -> clustering -> tilt model -> weekly report.
Finished months and evaluated positions are cached, so a weekly refresh only does new work.
"""

from __future__ import annotations

import argparse
import os

from tilt import blunder_model, cluster, download, label, parse, tilt_model, weekly_report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"))
    p.add_argument("--contact", default=os.environ.get("CHESSCOM_CONTACT"))
    p.add_argument("--username", default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--no-lstm", action="store_true")
    args = p.parse_args(argv)

    user = ["--username", args.username] if args.username else []
    steps = [
        ("download", lambda: download.main(user + (["--contact", args.contact] if args.contact else []))),
        ("parse", lambda: parse.main(user)),
        ("label", lambda: label.main(["--stockfish", args.stockfish])),
        ("blunder model", lambda: blunder_model.main([])),
        ("clustering", lambda: cluster.main([])),
        ("tilt model", lambda: tilt_model.main(["--no-lstm"] if args.no_lstm else [])),
        ("weekly report", lambda: weekly_report.main([])),
    ]
    for name, run in steps:
        if name == "download" and args.skip_download:
            continue
        print(f"\n########## {name} ##########")
        run()


if __name__ == "__main__":
    main()
