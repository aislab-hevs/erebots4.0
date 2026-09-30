# Architecture

Erebots4.0 is a framework for multi-agent systems in which agents are decoupled by
**protocols rather than imports**. This page explains the shape that follows
from that decision: what the layers are, what happens during one user turn, and
where the seams you extend at actually are.

## Three layers

```
  examples/            three frontends: Gradio, CLI, Vue SPA
  ─────────────────────────────────────────────────────────────
  bootstrap.py         composition: build the agents, wire the app
  app.py  demo.py      two deployments of the same framework
  specialists/         the example domain: the personal and conversion agents
  ─────────────────────────────────────────────────────────────
  erebots/             the framework — imports none of the above
```

The bottom layer is the installable package. **It never imports the layers above
it**, and that is enforced rather than promised: `tests/test_import_boundaries.py`
sweeps the package with an AST pass and fails if anything under `erebots/`
imports `specialists` or `gradio`. Every other claim on this page depends on
that one holding.

The middle layer is where an application lives. `bootstrap.py` builds the agents
and hands them to the framework's app factory; `specialists/registry.py` is the
single place local agents are declared. Neither the orchestrator nor the HTTP
surface knows what a "conversion agent" is.

## Two protocols, two seams

```
   Browser / CLI / SPA
          │  AgentEvent stream (SSE, or in-process)
          ▼
   ┌──────────────────┐   A2A (JSON-RPC, HTTP+JSON)   ┌───────────────────────┐
   │   Orchestrator   │ ────────────────────────────► │  Specialist agents     │
   │  routing + run() │                               │  /a2a/<name>           │
   └──────────────────┘                               └──────────┬────────────┘
          │  fallback: its own LLM                               │  MCP
          ▼                                                      ▼
    OpenAI-compatible LLM                          MCP tool servers /tools/<name>/mcp
```

**A2A** carries agent-to-agent traffic. A specialist is reached by URL, so it can
run in this process, in another container, or on another host in another
language — the orchestrator cannot tell the difference. What it knows about an
agent comes from that agent's card at `/.well-known/agent-card.json`: a name, a
description and a list of skills.

**MCP** carries agent-to-tool traffic. Deterministic capabilities are separate
servers, discovered at runtime rather than imported. See
[adding-a-tool.md](adding-a-tool.md).

Those two URLs are the extension points. Everything else in the diagram is
framework.

## One turn, end to end

A turn enters through `POST /api/chat/stream` (or, for an in-process frontend,
straight into `run()`), and leaves as a stream of events.

1. **Identity and validation.** `erebots/server/sse.py` validates the request
   body — this is the one place untrusted JSON becomes prompt material — and a
   session resolver derives the user from a cookie or a bearer token.
2. **Context.** `ConversationService` assembles what the turn gets to see: the
   user's stored profile rendered for prompting, and the chat history. Which
   history depends on who owns the transcript; see below.
3. **Routing.** The orchestrator asks its own LLM which specialists match,
   giving it each agent's name, description and — with `routing_include_skills`
   on — its skills. The plan is capped at `orchestrator_max_specialists`.
4. **Dispatch.** One specialist is called over A2A `message/stream`; several run
   concurrently when the request spans domains; none matching means the
   orchestrator answers directly from its own LLM.
5. **The specialist's own loop.** `BaseSpecialist` discovers its MCP tools on
   first use, then runs a bounded tool-calling loop: the model asks for a tool,
   the tool answers, the model continues. Tokens stream out as they arrive.
6. **Composition.** For a fan-out turn the contributions are merged into one
   reply — `multi_agent_composition` chooses between a synthesised answer and
   labelled sections — because several handoff sentences in a row confuse more
   than they help.
7. **Persistence.** The turn is appended to the user's history, unless the
   client owns the transcript.

Nothing in that path is UI-aware. The Gradio adapter, the CLI and the SPA each
consume the same events and render them differently.

## The event contract

`run()` yields `AgentEvent`s, defined in
[`erebots/runtime/events.py`](../erebots/runtime/events.py). Seven kinds:

| Type | Carries | Meaning |
|---|---|---|
| `routing` | `data["plan"]`, `data["target"]` | Which specialists were selected. |
| `token` | `text` | A visible answer chunk. |
| `reasoning` | `text` | A `<think>` chunk, for frontends that show reasoning. |
| `trace` | `trace_step` | One step of what happened — a routing decision, an A2A call, a tool invocation. |
| `input_required` | `text`, `data["task_id"]` | A specialist stopped to ask the user something. |
| `final` | `text`, `agent` | The answer, and who produced it. |
| `error` | `text` | The turn failed. |

Two fields make multi-agent turns renderable. `agent` names who the event came
from, and `channel` separates a contributing specialist's own stream (`side`)
from the answer being assembled (`answer`) — a frontend that ignores `channel`
behaves exactly as it did before fan-out existed.

## Asking the user a question

A specialist that cannot proceed without more information does not guess. It
**parks its A2A task** in `input-required` and the question streams to the user
as an `input_required` event carrying a `task_id`.

The frontend echoes `{"agent", "task_id"}` back as `context["pending"]` on the
next turn. The orchestrator then **skips routing entirely** and resumes that task
on that agent. This is deliberate: re-routing a bare answer like "38" through the
router could easily land it on a different specialist than the one that asked.

A handle that is stale — the task expired, the server restarted — or that belongs
to a different user falls back to normal routing rather than failing. Parked
tasks expire after `a2a_parked_task_ttl`.

## Where state lives

| State | Where | Notes |
|---|---|---|
| A2A tasks | In memory, or the database with `a2a_persistent_tasks=true` | Needs the database if more than one API instance serves the same users. |
| The user record | One `user_context` row per user | The framework defines the schema (`UserContextRow`), ships a store for it, and states the shape a replacement must have (`UserContextStoreProtocol`). Its own concern in that row is the conversation. |
| User profile and consents | The same row, through the agent that owns them | In the bundled example that is the personal agent and its `ProfileStore`. No other agent reads or writes them; one that needs them asks through that agent's tools. The framework offers only the column-merge primitive and does not know what a profile is. |
| Chat history | Either the store or the client — never both | See below. |
| Discovered MCP tools | Per specialist instance | Discovered on first use, retried on failure. |
| Agent registry | `agents.yaml` plus each agent's card | Read at startup; remote agents are just URLs. |

**Transcript ownership is one decision, made once.** A frontend that holds its
own transcript sends it, and that wins; one that holds nothing sends nothing and
gets the user's stored history. Never both, so no turn reaches the prompt twice.
Ownership decides persistence too: a client that sends its transcript owns it and
the service stores nothing, because a window of a conversation the client does
not fully hold would truncate the stored one. The `persisted` flag on
`GET /api/chat/history` reports which side is in charge, and it needs both halves
— an identity to store a conversation under *and* a store to put it in.

## Authentication

Two channels, two mechanisms, one dispatcher
(`erebots/api/auth_middleware.py`, internal plumbing rather than an extension
point):

- **Users** reach `/api/*` with a Keycloak session cookie or a bearer token.
  Both are validated the same way.
- **Machines** reach `/a2a/*` and `/tools/*` with a `client_credentials` token,
  checked on the `azp` claim.

The internal service account is fully trusted: A2A messages carry the acting
`user_id` in their metadata, so anything holding `internal_client_secret` can act
as any user. Treat it as a root credential.

CORS wraps authentication rather than the other way round, so a browser preflight
is answered and an unauthenticated 401 still carries its CORS headers — without
that the browser reports an opaque CORS failure instead of the authentication
error the user needs to see.

## Where to go next

- [adding-an-agent.md](adding-an-agent.md) — the A2A seam.
- [adding-a-tool.md](adding-a-tool.md) — the MCP seam.
- [api-stability.md](api-stability.md) — which import paths are supported.
- [`specialists/conversion/`](../specialists/conversion/README.md) — all of it,
  working, in 255 lines.
