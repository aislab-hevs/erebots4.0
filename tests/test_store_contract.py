"""Hold the shipped store to the protocol, and the protocol to the code.

The contract used to live only in a docstring, which said a store exposes
``save(ctx)``. No such method has ever existed in this project. Prose cannot
fail, so nothing contradicted it for as long as it stood.

Three checks here, in increasing order of what they would have caught:

1. the shipped store matches the protocol by name,
2. and by signature, which ``runtime_checkable`` does not verify,
3. and the protocol covers every call the service actually makes -- the one
   that closes the loop, because that is the direction the drift went.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from erebots.persistence import UserContext, UserContextStore
from erebots.runtime.contracts import UserContextLike, UserContextStoreProtocol

_SERVICE_SOURCE = Path(__file__).resolve().parents[1] / "erebots" / "runtime" / "conversation.py"

# Probed with getattr and fallen back from, so implementing it is optional.
# See the module docstring of erebots/runtime/contracts.py.
_OPTIONAL_ON_THE_STORE = {"append_chat_history"}

_PROTOCOL_METHODS = ("get_or_create", "set_chat_history")


def test_the_shipped_store_matches_the_protocol_by_name():
    # issubclass rather than isinstance: the store protocol carries only
    # methods, so the class answers, and no instance has to be built around a
    # database session this test would never use.
    assert issubclass(UserContextStore, UserContextStoreProtocol)


def test_the_shipped_context_matches_the_protocol_by_name():
    assert isinstance(UserContext(user_id="u1"), UserContextLike)


def test_a_partial_implementation_is_rejected():
    """Proves the check above can fail, which is the only thing that makes it worth running."""

    class HalfAStore:
        async def get_or_create(self, user_id: str, seed: dict | None = None): ...

    assert not issubclass(HalfAStore, UserContextStoreProtocol)


@pytest.mark.parametrize("name", _PROTOCOL_METHODS)
def test_the_store_and_the_protocol_agree_on_parameters(name: str):
    """``runtime_checkable`` checks names only, so the signatures are checked here.

    A store whose ``get_or_create`` took the seed under another name, or made it
    required, would satisfy ``isinstance`` and then fail at the first call.
    """
    declared = inspect.signature(getattr(UserContextStoreProtocol, name))
    implemented = inspect.signature(getattr(UserContextStore, name))

    assert [p.name for p in declared.parameters.values()] == [
        p.name for p in implemented.parameters.values()
    ], f"{name}: parameter names differ between the protocol and the store"

    assert [p.default for p in declared.parameters.values()] == [
        p.default for p in implemented.parameters.values()
    ], f"{name}: defaults differ between the protocol and the store"


def test_the_protocol_covers_every_call_the_service_makes():
    """The check that would have caught the ``save(ctx)`` fiction.

    Read from the source rather than by calling, because the interesting calls
    sit behind branches a unit test would have to contrive. A name that appears
    here and not in the protocol is either a contract that was never written
    down or a method nobody implements -- both were true at once, once.
    """
    source = _SERVICE_SOURCE.read_text(encoding="utf-8")
    called = set(re.findall(r"self\._store\.(\w+)", source))
    called |= set(re.findall(r'getattr\(\s*self\._store,\s*"(\w+)"', source))
    assert called, "no store calls found -- the pattern stopped matching the code"

    declared = {n for n in dir(UserContextStoreProtocol) if not n.startswith("_")}
    undeclared = called - declared - _OPTIONAL_ON_THE_STORE

    assert not undeclared, (
        f"ConversationService calls {sorted(undeclared)} on the store, which "
        "erebots.runtime.contracts does not declare. Add it to the protocol, or "
        "to the optional set here with the reason."
    )


def test_the_protocol_declares_nothing_the_service_never_calls():
    """The other direction: a promise nobody relies on is still a promise.

    Kept separate from the test above because the fix differs -- this one is
    removed from the protocol rather than added to it.
    """
    source = _SERVICE_SOURCE.read_text(encoding="utf-8")
    called = set(re.findall(r"self\._store\.(\w+)", source))
    called |= set(re.findall(r'getattr\(\s*self\._store,\s*"(\w+)"', source))

    unused = set(_PROTOCOL_METHODS) - called
    assert not unused, (
        f"the protocol declares {sorted(unused)}, which ConversationService "
        "never calls. Every method in it is a promise to implementers."
    )
