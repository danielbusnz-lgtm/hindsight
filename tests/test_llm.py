import pandas as pd

from hindsight.llm import Decision, Side, make_openai_decider


class _FakeCompletions:
    def __init__(self, side):
        self._side = side
        self.seen = None

    def parse(self, **kwargs):
        self.seen = kwargs
        parsed = Decision(side=self._side)
        message = type("M", (), {"parsed": parsed})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, side):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(side)})()


def test_side_maps_to_position():
    prices = pd.Series([100.0, 101.0, 102.0])
    assert make_openai_decider(client=_FakeClient(Side.long))(prices) == 1.0
    assert make_openai_decider(client=_FakeClient(Side.flat))(prices) == 0.0
    assert make_openai_decider(client=_FakeClient(Side.short))(prices) == -1.0


def test_lookback_trims_the_window():
    prices = pd.Series([float(i) for i in range(100)])
    client = _FakeClient(Side.flat)
    make_openai_decider(client=client, lookback=10)(prices)
    # Only the last 10 prices should reach the prompt.
    user_msg = client.chat.completions.seen["messages"][1]["content"]
    assert "99.00" in user_msg
    assert "89.00" not in user_msg
