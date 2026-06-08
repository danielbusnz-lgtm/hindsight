"""Append leaderboard results to a CSV so runs accumulate into a durable record.

One row per strategy per run, tagged with a UTC timestamp and the run config
(model, ticker, window, cutoff). The CSV is the source of truth a dashboard
(e.g. loomglass) can read later; the terminal print stays the quick look.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "run_at", "model", "ticker", "window", "cutoff", "strategy",
    "sharpe_pre", "sharpe_post", "ci_low", "ci_high", "p_value",
]


def append(path, scores, **meta) -> Path:
    """Append each Score as a row to the CSV at `path`, stamped with run metadata.

    `meta` carries the run config (model, ticker, window, cutoff). Extra keys are
    ignored, so callers can pass more context than the schema without breaking.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_header = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for s in scores:
            writer.writerow({
                "run_at": run_at,
                "strategy": s.name,
                "sharpe_pre": round(s.sharpe_pre, 4),
                "sharpe_post": round(s.sharpe_post, 4),
                "ci_low": round(s.ci_low, 4),
                "ci_high": round(s.ci_high, 4),
                "p_value": round(s.p_value, 4),
                **meta,
            })
    return path
