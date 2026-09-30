"""What `GenerativeLayer` puts in the request body.

`thinking=False` is the default, so the flag it produces rides on almost every
call the framework makes. It had no test at all, and that is how it came to be
spelled for one backend only: it disabled reasoning on vLLM and did nothing on
llama.cpp, where a one-word classification then cost 10.4s instead of 0.4s.
"""

from __future__ import annotations

import pytest

from erebots.agents.generative import GenerativeLayer

_MESSAGES = [{"role": "user", "content": "hi"}]


def _layer(*, thinking: bool) -> GenerativeLayer:
    return GenerativeLayer(
        api_key="EMPTY",
        api_base_url="http://x/v1",
        model="local",
        system_prompt="s",
        thinking=thinking,
    )


def test_disabling_thinking_is_spelled_for_both_backends():
    """Both spellings, or it works on one backend and silently not the other.

    vLLM reads the top-level key; llama.cpp reads `chat_template_kwargs`, which
    is what reaches the chat template. Neither is redundant — deleting either
    one restores the bug for that backend, without any test or error saying so.
    """
    body = _layer(thinking=False)._build_kwargs(_MESSAGES)["extra_body"]

    assert body["enable_thinking"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_an_agent_that_wants_reasoning_sends_no_flag():
    kwargs = _layer(thinking=True)._build_kwargs(_MESSAGES)

    assert "extra_body" not in kwargs


def test_the_parser_retry_drops_the_flag():
    """The retry exists because a backend choked on the request; send less."""
    kwargs = _layer(thinking=False)._build_kwargs(_MESSAGES, include_thinking_flag=False)

    assert "extra_body" not in kwargs


@pytest.mark.parametrize("thinking", [True, False])
def test_the_model_and_messages_are_always_there(thinking: bool):
    kwargs = _layer(thinking=thinking)._build_kwargs(_MESSAGES)

    assert kwargs["model"] == "local"
    assert kwargs["messages"] == _MESSAGES
