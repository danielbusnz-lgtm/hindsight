"""OpenAI decider: an LLM that sizes a position from recent prices.

Plugs into `harness.walk_forward` as the `decide` callable. It only ever sees
the price window handed to it, so it cannot peek at the future. The model is
forced to answer with a single conviction number via a structured-output
schema, so there is no free text to parse. The number is clamped to [-1, +1]
(fully short to fully long; 0 is flat). The backtester multiplies it straight
into returns, so a 0.5 means "half-sized long".

    from hindsight import harness, llm
    decide = llm.make_openai_decider(model="gpt-4o-mini")
    positions = harness.walk_forward(prices, decide)
"""

from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field


class Decision(BaseModel):
    conviction: float = Field(
        description="Position for the next bar, from -1 (max short) to +1 (max long); 0 is flat."
    )


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


_SYSTEM = (
    "You are a trading model. You are given recent prices for one asset, oldest "
    "first. Choose your position for the NEXT bar as a number from -1 to +1: "
    "+1 fully long, -1 fully short, 0 flat. Size it by how confident you are."
)


@lru_cache(maxsize=1)
def _default_client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY from the environment


def make_openai_decider(model: str = "gpt-4o-mini", lookback: int = 30, identify: bool = False, client=None):
    """Build a decider that asks an OpenAI model to size a position each bar.

    With identify=True the prompt names the ticker and current date (read from
    the price series' name and index). That identifiable context is what lets a
    model "recognise" a historical period and leak, so it's needed to measure
    leakage, but it does not let the decider see any future bar.
    """

    def decide(history: pd.Series) -> float:
        window = history.iloc[-lookback:]
        prices = ", ".join(f"{p:.2f}" for p in window)
        context = ""
        if identify:
            when = history.index[-1]
            when = when.date().isoformat() if hasattr(when, "date") else str(when)
            context = f"Asset: {history.name}. Current date: {when}.\n"
        completion = (client or _default_client()).chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"{context}Recent prices: {prices}\nYour position for the next bar?"},
            ],
            response_format=Decision,
        )
        parsed = completion.choices[0].message.parsed
        return _clamp(parsed.conviction) if parsed else 0.0

    return decide
