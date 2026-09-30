# A2A conformance

Erebots4.0 exposes each specialist as an A2A v1.0 service built on the official
[`a2a-sdk`](https://github.com/a2aproject/a2a-python). This page records **what
the agent cards declare** and **how a result against them has to be read**.
Results themselves are reported in the paper, against a stated commit. The
procedure for measuring it yourself is in [tck.md](tck.md); it is not repeated
here.

## What the cards declare

Each specialist's agent card at `/.well-known/agent-card.json` advertises:

- **Transports** — JSON-RPC and HTTP+JSON, served off the same request handler.
  gRPC is deliberately absent: it is not served, so it is not declared.
- **Methods** — `message/send`, `message/stream` (SSE), `tasks/get`,
  `tasks/list`, `tasks/cancel`, and the push-notification configuration CRUD.
- **Capabilities** — `streaming: true` (specialists stream token by token),
  `pushNotifications: true` (webhook delivery, validated against SSRF and DNS
  rebinding), and `extendedAgentCard: true` when an extended card is configured.
- **Skills** — from the agent's own skill registry.

This matters because **the TCK grades a server against its own card**. Declaring
less would raise the score and prove less; declaring something unimplemented
would fail immediately. The list above is what the implementation actually
serves.

## How it is measured

No counts here. Conformance results are reported in the paper, against a stated
commit and tool version; a second copy in the repository would be a second
number to keep in step, and the two have disagreed before.

What this page fixes is how such a number has to be read, because two of the
ways to misread it are built into the suite.

**Quote a count together with the endpoint it came from.** The two fixture
endpoints cover different requirements by construction: the ones only the
minimal card can exercise are `CORE-CAP-001/002/003`, which run only against a
server declaring the capability absent. A single endpoint's count and the union
are both correct and mean different things.

**The percentage is not the failure count.** a2a-tck scores
`PASS / (PASS + NOT TESTED)` and counts requirements *it ships no test for*
against the server under test. Skipped tests cost nothing; untested
requirements do. The failure count is what means something, and it is stable
across versions of the suite in a way the percentage is not.

To take a measurement, follow [tck.md](tck.md) and fill in its run record.

## Why two endpoints

`/a2a/tck` declares every capability. `/a2a/tck-minimal` serves the same
scenarios behind a card declaring `streaming`, `pushNotifications` and
`extendedAgentCard` all false.

Both are needed, and neither is sufficient. Some requirements pin the error a
server owes a caller that asks for a capability it does *not* have — those can
only run against the minimal card. Their positive counterparts can only run
against the full one. A positive test and its negative can never both run
against the same card, so the union of the two runs is what covers the
requirements.

The fixtures are mounted only when `a2a_conformance_agent=true`. They serve
canned payloads and belong to a test run, not to a deployment.

## Two deviations worked around here

Both are `a2a-sdk` behaviours that the spec contradicts, corrected in
`erebots/agents/a2a/server.py` and marked for removal once an SDK release fixes
them upstream:

- **`CORE-CANCEL-002`** — `TaskNotCancelableError` mapped to HTTP 400 where the
  spec's error table gives it 409.
- **`STREAM-SUB-003`** — subscribing to a terminal task answered
  `InvalidParamsError` (-32602) instead of `UnsupportedOperationError` (-32004).
  The SDK's v1 handler got this right; its v2 replacement does not.

## Notes

- Point the TCK at a **specialist** endpoint. The orchestrator is not exposed
  over A2A — it is the *client* that routes to specialists.
- This needs a live server and an external tool, so it is a manual procedure.
  The in-process round-trip tests (`tests/test_a2a_server.py`) cover the same
  code paths and do run under `uv run pytest`.
