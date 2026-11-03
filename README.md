# Erebots4.0

**A protocol-first framework for building multi-agent AI systems.**

A generalist *orchestrator* routes each user request to the right domain
*specialist* — and agents are decoupled by open **protocols**, not by Python
imports:

- **A2A** ([Agent2Agent](https://a2a-protocol.org), v1.0) between agents — a
  specialist is just a URL. It can run in this process, in another container, or
  on another server in any language.
- **MCP** ([Model Context Protocol](https://modelcontextprotocol.io)) for tools
  — deterministic capabilities are separate MCP servers, discovered at runtime.

The core is **headless and frontend-agnostic**: it emits a neutral event stream,
so a Gradio UI, a CLI, and a browser SPA all drive the *same* orchestrator. The
bundled example pairs a personal agent, which owns the user's profile, with a
unit-conversion specialist that answers from arithmetic rather than from a
model's arithmetic.

---

## Why Erebots4.0

| | |
|---|---|
| **Open protocols, not glue** | Agents talk A2A; tools talk MCP. Swap, relocate, or add agents without touching the framework. |
| **Frontend-agnostic core** | `orchestrator.run()` yields UI-neutral `AgentEvent`s. Ship a chat UI, a CLI, or your own React/Vue app on the same backend. |
| **Standard streaming (SSE)** | One `POST /api/chat/stream` SSE endpoint — the transport the whole stack (A2A, MCP, LLM APIs) already speaks. |
| **Auth built in** | Keycloak OIDC for users (cookie *or* bearer) and machine-to-machine `client_credentials` for the internal `/a2a/*` and `/tools/*` channels. |
| **Any OpenAI-compatible LLM** | llama.cpp, vLLM, Ollama, or a hosted API — set a base URL. |
| **Framework, not app** | The core (`erebots/`) never imports the example domain. Adding an agent is application code + one line of config, guarded by an import-boundary test. |

## Architecture

```
   Browser / CLI / SPA
          │  AgentEvent stream (SSE or in-process)
          ▼
   ┌──────────────────┐   A2A (JSON-RPC)   ┌──────────────────────┐
   │   Orchestrator   │ ─────────────────► │  Specialist agents    │
   │  routing + run() │                    │  /a2a/<name>          │
   └──────────────────┘                    └──────────┬───────────┘
          │  fallback (own LLM)                       │  MCP
          ▼                                           ▼
      OpenAI-compatible LLM                    MCP tool servers /tools/<name>

   Keycloak (OIDC + M2M)   ·   Postgres (user context / history)
```

How the pieces fit together, and what happens during one turn:
[docs/architecture.md](docs/architecture.md). Everything else is indexed in
[docs/](docs/README.md).

- `erebots/` — the reusable **framework** (agents, A2A/MCP clients+servers,
  orchestrator, runtime events, SSE server, security, persistence).
- `bootstrap.py` + `app.py` + `agents.yaml` — the **example app** composition
  (what to build, mount, and route to).
- `specialists/` — the **example domain** (the personal and conversion specialists).

## Quickstart (local, no Docker)

You need [uv](https://docs.astral.sh/uv/) and any OpenAI-compatible LLM running
(e.g. `llama.cpp` on `http://127.0.0.1:8080`). No model yet?
[docs/running-a-model.md](docs/running-a-model.md) sets one up, on a
workstation or a laptop.

```bash
uv sync
MODEL=local   # or the id from: curl -s http://127.0.0.1:8080/v1/models
LLM=http://127.0.0.1:8080/v1

# Export once: every command below needs the same configuration. Without it the
# defaults apply — authentication on, Postgres — and the app refuses to start.
# A `.env` file in the working directory does the same job and survives the shell.
export enable_auth=false
export database_url="sqlite+aiosqlite:///./dev.db"
export orchestrator_api_base_url=$LLM conversion_api_base_url=$LLM personal_api_base_url=$LLM
export orchestrator_model=$MODEL conversion_model=$MODEL personal_model=$MODEL

uv run uvicorn app:app --host 127.0.0.1 --port 7860
```

Open **http://127.0.0.1:7860** for the Gradio chat, or talk to the headless core
directly — in the same shell, so it inherits the configuration above:

```bash
echo "How many pounds is 70 kg?" | uv run python examples/cli_chat.py

# ...or as a known user: its profile, consents and history come from the store,
# so a later run continues the same conversation.
uv run python examples/cli_chat.py --user alice
```

That first one prints `[routing → conversion]` and then `70 kg = 154.324 lb`.
Both halves matter: the orchestrator picked a specialist rather than answering
itself, and the number came from a deterministic MCP tool rather than from the
model's arithmetic. A slightly different number means the tool never ran.

For the full stack with Keycloak auth and Postgres, see
[docs/deployment.md](docs/deployment.md) (`docker compose up --build`).

### Minimal demonstrator (no Gradio, no RAG)

[`demo.py`](demo.py) runs the same protocol path — routing, A2A, MCP tools,
persistence — with two specialists: the **personal** agent, the only one allowed
to touch a user's profile and consents, and the **conversion** agent (unit
conversion through a deterministic MCP tool). It imports no Gradio, so it
installs from the light core:

```bash
uv sync --no-default-groups --group dev --frozen   # core only: no gradio
enable_auth=false \
database_url="sqlite+aiosqlite:///./demo.db" \
agents_config_path=agents.demo.yaml \
conversion_api_base_url=http://127.0.0.1:8080/v1 \
conversion_model=$MODEL \
uv run --no-default-groups --group dev --frozen \
  uvicorn demo:app --host 127.0.0.1 --port 7860
```

The group selection has to be repeated on `uv run`: it re-resolves the
environment on every invocation, so a bare `uv run` would reinstall the default
group (Gradio) that the `uv sync` above deliberately left out, and the demo
would no longer be evidence of a light install.

It serves the Vue SPA at **http://127.0.0.1:7860** and the same
`POST /api/chat/stream`. The Gradio UI is an optional extra
(`erebots[gradio]`), installed by default in a checkout.

## Frontends (same core, three ways)

| Frontend | Where | Consumes |
|---|---|---|
| **Gradio** chat UI | `examples/gradio_ui/` | `run()` in-process |
| **CLI** | [`examples/cli_chat.py`](examples/cli_chat.py) | `run()` in-process, `--user` for a stored identity |
| **Web SPA** (Vue) | [`examples/web/`](examples/web/) | `POST /api/chat/stream` (SSE), `GET /api/chat/history` |

The web example is a single no-build HTML file with Vue from a CDN; the same
pattern works in React.

All three go through the same `ConversationService`, so identity, history and
persistence mean the same thing in each. A frontend that owns the transcript
sends it and that wins; one that keeps no state sends nothing and gets the
user's stored history — never both, so no turn reaches the prompt twice.

Ownership decides persistence too. A client that sends its transcript owns it
and the service stores nothing (a window of a conversation the client does not
fully hold would truncate the stored one). A client that sends nothing gets the
turn appended for it, and reads the conversation back from
`GET /api/chat/history` on load — which is what lets the SPA continue a
conversation after a reload, or on another device. That endpoint's `persisted`
flag says which side is in charge. It takes both halves: an identity to store a
conversation under *and* a store to put it in. Missing either, it is `false` and
the client has to keep its own.

## Adding an agent

Adding a specialist touches **only application code — never `erebots/`**:

1. Subclass `BaseSpecialist` in your domain (system prompt, skills, MCP tools).
2. Declare it in `specialists/registry.py`.
3. Route to it: add its A2A URL to `agents.yaml`.

An **autonomous agent on another host** needs only step 3 — just its URL. See
[docs/adding-an-agent.md](docs/adding-an-agent.md), and
[docs/adding-a-tool.md](docs/adding-a-tool.md) for the MCP side.

The worked example is
[`specialists/conversion/`](specialists/conversion/README.md): a complete
specialist in 255 lines, answering from arithmetic rather than from a model's
arithmetic.

## Status and limitations

**Erebots4.0** is the project; **1.0.0** is this release of it. The name carries
the generation of the research platform, the version carries the compatibility
promise below, and the Python package you install and import is plain `erebots`.
The three are easy to run together, so they are separated here once.

It is developed in a research lab and published because the protocol work is
worth reusing. What the version commits to, and what it does not:

- **A supported import path breaks only in a major release** — with four
  exceptions, named rather than implied, in
  [docs/api-stability.md](docs/api-stability.md). Every break is a
  [CHANGELOG.md](CHANGELOG.md) entry naming its replacement. The table of
  supported paths is deliberately short, because every row is a promise.
- **A2A v1.0 conformance is measured rather than claimed**, and the procedure is
  reproducible: [docs/tck.md](docs/tck.md) runs the official suite against a live
  endpoint. Results are reported in the paper, against a stated commit.
  [docs/a2a-conformance.md](docs/a2a-conformance.md) has what the cards declare
  and the two deviations worked around here rather than fixed upstream.
- **The bundled Keycloak realm is a development fixture.** Its credentials are
  documented placeholders, the app refuses to start on them with authentication
  enabled, and it ships no signing key — Keycloak generates one on import.
- **Two settings relax security and are off by default**, both listed as
  accepted risks in [SECURITY.md](SECURITY.md) and documented in
  [docs/configuration.md](docs/configuration.md).
- **The example domain is an example.** `specialists/` and the applications on
  top of it exist to demonstrate the framework, not to be deployed.


Known gaps, in the order they matter: when several specialists in one fan-out turn
each stop to ask the user something, only one of those questions can be answered
back and the rest expire; and the project targets Python 3.13 only.

## Development

```bash
uv run pytest          # test suite
uvx ruff check .       # lint
```

The suite includes an **import-boundary test** that fails if the framework ever
imports the example domain.

The same two commands run on every push and pull request, on Linux and Windows,
alongside a job that installs the built wheel into an empty environment and
imports every supported path — see
[.github/workflows/ci.yml](.github/workflows/ci.yml). Nothing there needs a
secret or a service, so it runs unchanged on a fork.

Which import paths the framework supports, what the promise covers, and the four
paths deliberately carved out of it: [docs/api-stability.md](docs/api-stability.md).

## Getting help

- **Questions and bugs** — [open an issue](https://github.com/aislab-hevs/erebots4.0/issues).
- **Security** — do not open an issue. See [SECURITY.md](SECURITY.md); reports go
  privately to elia.pacioni@hevs.ch.
- **Source** — <https://github.com/aislab-hevs/erebots4.0>

## Citing

If you use Erebots4.0 in academic work, cite it with our preprint: 
Pacioni, E., Calvaresi, D., Coman, A. C., Jaggi, I., Crocetti, V., Manzo, G., & Schumacher, M. I. (2026). EREBOTS 4.0: A framework for conversational assistants with A2A agents and MCP tools. arXiv preprint arXiv:2610.36221.

[AUTHORS](AUTHORS) carries the same list in plain text, with each author's ORCID.

## License

[MIT](LICENSE) © 2026 HES-SO Valais-Wallis, Applied Intelligent Agents Lab (AISLab)

Third-party material — the text the code of conduct is adapted from, and what the
web example loads at runtime — is recorded in [NOTICE](NOTICE), together with a
licence review of the dependencies.
