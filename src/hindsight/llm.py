"""LLM deciders: models that size a position from recent prices.

Each `make_*_decider` plugs into `harness.walk_forward` as the `decide`
callable. It only ever sees the price window handed to it, so it cannot peek
at the future. The model is forced to answer with a single conviction number
via a structured-output schema, so there is no free text to parse. The number
is clamped to [-1, +1] (fully short to fully long; 0 is flat). The backtester
multiplies it straight into returns, so a 0.5 means "half-sized long".

    from hindsight import harness, llm
    decide = llm.make_openai_decider(model="gpt-4o-mini")
    decide = llm.make_anthropic_decider(model="claude-opus-4-8")
    positions = harness.walk_forward(prices, decide)

`make_decider` picks the right factory from the model's provider in the
cutoff registry, so callers can iterate a model list without caring who
serves what.
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
        prompt = _user_prompt(history, lookback, identify)
        completion = (client or _default_client()).chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format=Decision,
        )
        parsed = completion.choices[0].message.parsed
        return _clamp(parsed.conviction) if parsed else 0.0

    return decide


def _user_prompt(history: pd.Series, lookback: int, identify: bool) -> str:
    """The shared per-bar prompt: identical across providers so a cross-model
    comparison measures the model, not the prompt."""
    window = history.iloc[-lookback:]
    prices = ", ".join(f"{p:.2f}" for p in window)
    context = ""
    if identify:
        when = history.index[-1]
        when = when.date().isoformat() if hasattr(when, "date") else str(when)
        context = f"Asset: {history.name}. Current date: {when}.\n"
    return f"{context}Recent prices: {prices}\nYour position for the next bar?"


@lru_cache(maxsize=1)
def _default_anthropic_client():
    import anthropic

    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def make_anthropic_decider(
    model: str = "claude-opus-4-8", lookback: int = 30, identify: bool = False, client=None
):
    """Build a decider that asks a Claude model to size a position each bar.

    Same contract and prompt as `make_openai_decider`. No sampling parameters:
    Opus 4.7+ rejects `temperature`, and the structured-output schema already
    pins the response shape.
    """

    def decide(history: pd.Series) -> float:
        prompt = _user_prompt(history, lookback, identify)
        response = (client or _default_anthropic_client()).messages.parse(
            model=model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=Decision,
        )
        parsed = response.parsed_output
        return _clamp(parsed.conviction) if parsed else 0.0

    return decide


def make_decider(model: str, lookback: int = 30, identify: bool = False, client=None):
    """Build a decider for `model`, dispatching on the provider recorded in the
    cutoff registry. Raises KeyError for a model with no registry entry, which
    is deliberate: a model without a known cutoff cannot be graded honestly."""
    from hindsight import cutoffs

    provider = cutoffs.provider(model)
    factory = {"openai": make_openai_decider, "anthropic": make_anthropic_decider}[provider]
    return factory(model=model, lookback=lookback, identify=identify, client=client)
