import csv

from hindsight import results
from hindsight.evaluate import Score


def test_append_accumulates_and_writes_header_once(tmp_path):
    path = tmp_path / "runs" / "leaderboard.csv"
    score = Score("gpt-4o-mini", 0.21, 0.19, -0.06, 0.41, 0.14)
    meta = dict(model="gpt-4o-mini", ticker="AAPL", window="w", cutoff="2023-10-01")

    results.append(path, [score], **meta)
    results.append(path, [score], **meta)

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2                      # both runs accumulate
    assert rows[0]["strategy"] == "gpt-4o-mini"
    assert rows[0]["model"] == "gpt-4o-mini"
    assert rows[0]["sharpe_post"] == "0.19"
    # Header written once: only one line equals the header text.
    assert path.read_text().count("run_at,model") == 1
