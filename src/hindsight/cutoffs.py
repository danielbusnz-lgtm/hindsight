"""Model knowledge-cutoff registry.

Provider-stated cutoffs are approximate FLOORS, not guarantees: a model often
knows events after its stated date (training contamination, RLHF on fresher
data). Treat the post-cutoff window as best effort, add a buffer if you want to
be safe, and prefer inferring the real boundary from the leakage curve when you
can. Data lives in data/model_cutoffs.json; edit it there.
"""

import json
from pathlib import Path

import pandas as pd

_PATH = Path(__file__).resolve().parents[2] / "data" / "model_cutoffs.json"
_REGISTRY = json.loads(_PATH.read_text())["models"]


def cutoff(model: str) -> pd.Timestamp:
    """Stated knowledge cutoff for one model."""
    if model not in _REGISTRY:
        raise KeyError(f"no cutoff on record for {model!r}; add it to {_PATH.name}")
    return pd.Timestamp(_REGISTRY[model]["cutoff"])


def common_cutoff(models) -> pd.Timestamp:
    """Latest cutoff across models: the start of the window they were all blind to.

    Use this for a fair cross-model bake-off, grade every candidate only on data
    after this date, so no model in the lineup could have memorised any of it.
    """
    return max(cutoff(m) for m in models)
