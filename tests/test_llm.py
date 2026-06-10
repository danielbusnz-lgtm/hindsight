import pandas as pd

from hindsight.llm import Decision, make_anthropic_decider, make_decider, make_openai_decider


class _FakeCompletions:
    def __init__(self, conviction):
        self._conviction = conviction
        self.seen = None

    def parse(self, **kwargs):
        self.seen = kwargs
        parsed = Decision(conviction=self._conviction)
        message = type("M", (), {"parsed": parsed})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, conviction):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(conviction)})()


def test_conviction_passes_through():
    prices = pd.Series([100.0, 101.0, 102.0])
    assert make_openai_decider(client=_FakeClient(0.5))(prices) == 0.5
    assert make_openai_decider(client=_FakeClient(-0.25))(prices) == -0.25
    assert make_openai_decider(client=_FakeClient(0.0))(prices) == 0.0


def test_conviction_is_clamped():
    prices = pd.Series([100.0, 101.0, 102.0])
    assert make_openai_decider(client=_FakeClient(1.7))(prices) == 1.0
    assert make_openai_decider(client=_FakeClient(-3.0))(prices) == -1.0


def test_lookback_trims_the_window():
    prices = pd.Series([float(i) for i in range(100)])
    client = _FakeClient(0.0)
    make_openai_decider(client=client, lookback=10)(prices)
    # Only the last 10 prices should reach the prompt.
    user_msg = client.chat.completions.seen["messages"][1]["content"]
    assert "99.00" in user_msg
    assert "89.00" not in user_msg


class _FakeMessages:
    def __init__(self, conviction):
        self._conviction = conviction
        self.seen = None

    def parse(self, **kwargs):
        self.seen = kwargs
        parsed = Decision(conviction=self._conviction)
        return type("R", (), {"parsed_output": parsed})()


class _FakeAnthropicClient:
    def __init__(self, conviction):
        self.messages = _FakeMessages(conviction)


def test_anthropic_conviction_passes_through_and_clamps():
    prices = pd.Series([100.0, 101.0, 102.0])
    assert make_anthropic_decider(client=_FakeAnthropicClient(0.5))(prices) == 0.5
    assert make_anthropic_decider(client=_FakeAnthropicClient(1.7))(prices) == 1.0
    assert make_anthropic_decider(client=_FakeAnthropicClient(-3.0))(prices) == -1.0


def test_anthropic_lookback_trims_the_window():
    prices = pd.Series([float(i) for i in range(100)])
    client = _FakeAnthropicClient(0.0)
    make_anthropic_decider(client=client, lookback=10)(prices)
    user_msg = client.messages.seen["messages"][0]["content"]
    assert "99.00" in user_msg
    assert "89.00" not in user_msg


def test_anthropic_sends_no_sampling_params():
    # Opus 4.7+ rejects temperature/top_p/top_k; the decider must not send them.
    prices = pd.Series([100.0, 101.0, 102.0])
    client = _FakeAnthropicClient(0.0)
    make_anthropic_decider(client=client)(prices)
    for param in ("temperature", "top_p", "top_k"):
        assert param not in client.messages.seen


def test_make_decider_dispatches_on_provider():
    prices = pd.Series([100.0, 101.0, 102.0])
    assert make_decider("gpt-4o-mini", client=_FakeClient(0.25))(prices) == 0.25
    assert make_decider("claude-opus-4-8", client=_FakeAnthropicClient(0.25))(prices) == 0.25


def test_make_decider_rejects_unregistered_model():
    try:
        make_decider("not-a-real-model")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for a model with no cutoff on record")
