"""Pull daily closing prices into data/prices.csv.

A refresh tool, not a library dependency. Span chosen to straddle gpt-4o-mini's
~Oct-2023 training cutoff, so there's a pre-cutoff (maybe remembered) stretch
and a post-cutoff (provably blind) stretch.

    python scripts/fetch_prices.py
"""

from pathlib import Path

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
START = "2022-01-01"
END = "2026-06-01"
OUT = Path(__file__).resolve().parents[1] / "data" / "prices.csv"


def main():
    data = yf.download(TICKERS, start=START, end=END, interval="1d", auto_adjust=True, progress=False)
    close = data["Close"].dropna(how="any")  # keep only dates every ticker trades
    OUT.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(OUT)
    print(f"saved {close.shape[0]} rows x {close.shape[1]} tickers -> {OUT}")
    print(close.tail(3).round(2))


if __name__ == "__main__":
    main()
