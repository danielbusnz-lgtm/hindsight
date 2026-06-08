"""OpenAI decider: an LLM that picks long / flat / short from recent prices.

Plugs into `harness.walk_forward` as the `decide` callable. It only ever sees
the price window handed to it, so it cannot peek at the future. The model is
forced to answer with a single side via a structured-output schema, so there
is no free text to parse.

    from hindsight import harness, llm
    decide = llm.make_openai_decider(model="gpt-4o-mini")
    positions = harness.walk_forward(prices, decide)
"""

from enum import Enum
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel


class Side(str, Enum):
    long = "long"
    flat = "flat"
    short = "short"


class Decision(BaseModel):
    side: Side


_TO_POSITION = {Side.long: 1.0, Side.flat: 0.0, Side.short: -1.0}

_SYSTEM = (
    "You are a trading model. You are given recent prices for one asset, oldest "
    "first. Decide your position for the NEXT bar: long if you expect the price "
    "to rise, short if you expect it to fall, flat if unsure. Answer with one side."
)


@lru_cache(maxsize=1)
def _default_client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY from the environment


def make_openai_decider(model: str = "gpt-4o-mini", lookback: int = 30, client=None):
    """Build a decider that asks an OpenAI model for a position each bar."""

    def decide(history: pd.Series) -> float:
        window = history.iloc[-lookback:]
        prices = ", ".join(f"{p:.2f}" for p in window)
        completion = (client or _default_client()).chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Recent prices: {prices}\nYour position for the next bar?"},
            ],
            response_format=Decision,
        )
        parsed = completion.choices[0].message.parsed
        return _TO_POSITION[parsed.side] if parsed else 0.0

    return decide
