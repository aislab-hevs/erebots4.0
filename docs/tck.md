# Running the A2A TCK

Reproducible procedure for verifying that Erebots4.0's A2A endpoints actually
implement what their agent cards declare, using the official
[a2a-tck](https://github.com/a2aproject/a2a-tck).

The TCK turns *"A2A v1.0 conformant"* from a claim into a measurement: it fetches
the agent card, then exercises every declared method and capability against the
spec. Only once it passes should the README make that claim — see
[a2a-conformance.md](a2a-conformance.md) for what we declare and why.

This is a **manual** procedure: it needs a live server and an external tool, so
it is not part of the unit suite. The in-process round-trip tests
(`tests/test_a2a_server.py`) cover the same code paths and do run under
`uv run pytest`.

## Prerequisites

| Component | Requirement |
|---|---|
| Erebots4.0 | Python 3.13, `a2a-sdk` 1.1.2 |
| a2a-tck | Cloned separately, own venv (Python 3.14 works) |
| LLM | Any OpenAI-compatible endpoint, reachable |

You need a real LLM: the TCK calls `message/send` for real, so the specialist
will invoke the model. Without one the tests time out.

## 1. Start the LLM

Any OpenAI-compatible server (llama.cpp, vLLM, Ollama). Note the port and the
model id:

```bash
curl -s http://127.0.0.1:3960/v1/models | jq '.data[].id'
```

## 2. Start Erebots4.0 with authentication disabled

> **This step is mandatory.** With `enable_auth=true`, `/a2a/*` requires a
> service-account bearer token (see `erebots/api/auth_middleware.py`). The TCK
> has no such token and every single test would fail with `401`.

```bash
cd /path/to/erebots

LLM=http://127.0.0.1:3960/v1
MODEL=Qwen3.6-35B-A3B          # from step 1

enable_auth=false \
database_url="sqlite+aiosqlite:///./dev.db" \
a2a_conformance_agent=true \
a2a_push_allow_insecure_webhooks=true \
orchestrator_api_base_url=$LLM \
conversion_api_base_url=$LLM \
orchestrator_model=$MODEL conversion_model=$MODEL \
uv run --frozen uvicorn app:app --host 127.0.0.1 --port 7860
```

`a2a_conformance_agent=true` mounts the endpoint the suite drives (`/a2a/tck`);
`a2a_push_allow_insecure_webhooks=true` is required because the suite's push
receiver listens on `http://localhost`, which the default webhook policy
(https + globally routable) would reject at registration.

**Do not use `docker compose` for this.** The compose stack reaches the API only
through Traefik on `https://app.erebots.local` — `erebots-api` publishes no host
port, TLS uses mkcert certificates, and HTTP is redirected to HTTPS. All three
get in the TCK's way, and with authentication disabled Keycloak and Postgres
serve no purpose. The local uvicorn run above gives plain HTTP on loopback,
which is exactly what the TCK expects.

## 3. Verify the agent card

Before installing anything, confirm the endpoint is reachable. If this fails,
the TCK cannot start:

```bash
curl -s http://127.0.0.1:7860/a2a/tck/.well-known/agent-card.json | jq
```

Expected: two `supportedInterfaces` (`JSONRPC` and `HTTP+JSON`),
`protocolVersion: 1.0`, and `capabilities` declaring `streaming`,
`pushNotifications` and `extendedAgentCard` all `true`.

## 4. Install the TCK

Clone it **outside** the Erebots4.0 working tree, with its own venv — installing it
inside the project risks touching `uv.lock`.

```bash
cd ~/GitHub                  # anywhere except inside erebots/
git clone https://github.com/a2aproject/a2a-tck.git
cd a2a-tck
uv venv
uv pip install -e .
```

## 5. Run it

> **Invoke the venv interpreter explicitly.** `run_tck.py` builds its pytest
> command from `sys.executable`; running it as `./run_tck.py` without an
> activated venv resolves the shebang to the system Python and fails with
> `No module named pytest`. Calling `.venv/bin/python` directly works from any
> shell and needs no activation.

**First pass — MUST only.** These are the blocking requirements; this is the run
that decides whether the conformance claim holds:

```bash
cd ~/GitHub/a2a-tck
.venv/bin/python run_tck.py \
    --sut-host http://127.0.0.1:7860/a2a/tck \
    --transport jsonrpc \
    --level must -v
```

**Second pass — full run**, to populate the conformance table with SHOULD and
MAY results as well. Note the **absence** of `--transport`: this is the run the
record at the bottom of this page reports.

```bash
.venv/bin/python run_tck.py \
    --sut-host http://127.0.0.1:7860/a2a/tck -v
```

`--transport jsonrpc` restricts the suite to that one binding, which roughly
halves the parametrized tests — 102 passed against 180 for the same server.
Both runs are valid; they measure different things, and only the unrestricted
one covers the HTTP+JSON binding the card also declares. Do not compare their
totals.

**Third pass — the capability-less twin.** `CORE-CAP-001/002/003` pin the error
a server owes a caller asking for a capability it does not have, so the suite
skips them against a card that has it: they are unreachable on `/a2a/tck` by
construction. `/a2a/tck-minimal` serves the same scenarios behind a card
declaring `streaming`, `pushNotifications` and `extendedAgentCard` all false,
which is where those three run:

```bash
.venv/bin/python run_tck.py \
    --sut-host http://127.0.0.1:7860/a2a/tck-minimal
```

Read it together with the main run, not instead of it: this one necessarily
skips every test the capabilities gate, and the main one skips their absence.
Neither number means much alone — the union is what covers the requirements.

Conformance is measured per endpoint, so repeat both against every specialist
a deployment serves. This application ships one:

```bash
.venv/bin/python run_tck.py \
    --sut-host http://127.0.0.1:7860/a2a/conversion \
    --transport jsonrpc --level must -v
```

## 6. Collect the reports

Always written to `a2a-tck/reports/`:

| File | Use |
|---|---|
| `compatibility.json` | Machine-readable — the source for the conformance table |
| `compatibility.html` | Readable summary; use it to diagnose failures |
| `tck_report.html` | Standard pytest report |
| `junitreport.xml` | CI integration, if this is ever automated |

## Interpreting the results

The TCK classifies tests by RFC 2119 level:

- **MUST** — absolute requirements. Any failure breaks the conformance claim and
  must be fixed or the claim narrowed.
- **SHOULD** — expected behaviour, reported as `xfail`. Failures are acceptable;
  report them as known deviations rather than hiding them.
- **MAY** — optional. Skipped when the card does not declare the capability.
  Note the mirror case too: tests of the *absence* of a capability (e.g.
  "push not supported returns -32003") are skipped precisely because the
  capability is on — a positive test and its negative can never both run
  against the same card, so 100% with such skips is the ceiling for one run.

Record the outcome as a capability-declared → verified table:

| Declared in the agent card | TCK result |
|---|---|
| Transports: JSON-RPC, HTTP+JSON | |
| `message/send` | |
| `message/stream` (SSE) | |
| `tasks/get` / `tasks/list` / `tasks/cancel` | |
| `streaming: true` | |
| `pushNotifications: true` (CRUD + delivery) | |
| `extendedAgentCard: true` | |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named pytest` | Ran `./run_tck.py` without the venv active; shebang resolved to system Python | Use `.venv/bin/python run_tck.py …` |
| `401` on every test | `enable_auth=true` — `/a2a/*` demands a service-account bearer | Restart Erebots4.0 with `enable_auth=false` |
| Connection refused | Server not up, or reached via Docker/Traefik | Use the local uvicorn command in step 2 |
| Tests hang or time out | LLM unreachable; `message/send` never completes | Check the `*_api_base_url` ports against step 1 |
| TLS / certificate errors | Pointing at the Traefik HTTPS endpoint | Use plain HTTP on `127.0.0.1:7860` |

## Run record

Fill this in for every run that is cited in a publication. A measurement is only
worth as much as the versions it was taken against, which is why the commit and
both tool versions are fields rather than prose.

| Field | Value |
|---|---|
| Date | |
| Erebots4.0 commit | |
| `a2a-sdk` version | |
| a2a-tck commit | |
| LLM / model | none — the fixtures answer from canned payloads, no model in the loop |
| Command | |
| Endpoints tested | |
| Tests | |
| Requirements on `/a2a/tck` | |
| Requirements on `/a2a/tck-minimal` | |
| Requirements, union of both | |
| Percentages on `/a2a/tck` | |

## What a run will not reach, by design

Two groups of requirements stay unreached however the suite is run, and both are
choices rather than gaps. Recording them here means a later run does not have to
rediscover why its score is not 100%.

The `GRPC-*` requirements: no gRPC binding is declared. And `CARD-EXT-002`,
which measures a server declaring the extended-card capability without having
configured one — `_build_card` derives that capability from the configuration
precisely so the state cannot occur, so reaching the requirement would mean
shipping an endpoint broken on purpose.

**Read a percentage together with the counts.** a2a-tck scores
`PASS / (PASS + NOT TESTED)` and counts requirements it ships no test for
(`CARD-SIGN-*`, `AUTH-*`, `VER-CLIENT-*`, `BIND-EQUIV-*`, `GRPC-SVC-003`)
against the server. Since a2a-tck `cf4985b` those count against the score
instead of being excluded, which is why an earlier run of the *same* server
scored higher without anything about the deployment changing.

## Two deviations found in the SDK

Two MUST requirements regressed on the 2026-09-01 suite, both because
a2a-tck `de6af18` began asserting the specific error a requirement names
instead of accepting any error, and both traced to `a2a-sdk` 1.1.2 answering
something other than the spec:

- `CORE-CANCEL-002` — `TaskNotCancelableError` mapped to HTTP 400 rather than
  the 409 the spec's error table gives it.
- `STREAM-SUB-003` — subscribing to a terminal task answered
  `InvalidParamsError` (-32602) instead of `UnsupportedOperationError`
  (-32004); the SDK's v1 handler got this right, its v2 replacement does not.

Both are corrected in `erebots/agents/a2a/server.py` (the `A2A_ERROR_MAPPING`
entry and `HistoryPreservingHandler.on_subscribe_to_task`). Remove the
workarounds once an SDK release fixes them upstream.

Three earlier ones, against `a2a-sdk` 1.1.0, are fixed in the same file and
listed because a reader of that code needs to know why the workarounds are
there: `DM-SERIAL-005` (strict rejection of unknown fields) and
`CORE-HIST-005/006` (follow-up messages dropped from task history by the SDK's
id-only dedup when a client reuses message ids), handled by `LenientJsonBodies`
and `HistoryPreservingHandler`.

Each workaround is a claim about a specific SDK version. Re-check them when the
dependency moves, and take the measurement with the run record above.
