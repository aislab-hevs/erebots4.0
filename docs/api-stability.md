# Public API and stability

**Erebots4.0** is **1.0.0**. This page says which import paths of the `erebots`
package are supported, which are internal, and what "supported" is worth now that
the number promises something.

The public API is a set of **documented module paths**, not one flat top-level
namespace. `import erebots` gives you `__version__` and nothing else, on purpose:
the framework's parts have different dependencies and different import costs, and
collapsing them into one namespace would make every consumer pay for all of them.

The package ships a `py.typed` marker, so a type checker reads the inline
annotations of everything below.

## Supported

| Import path | What it is for |
|---|---|
| `erebots.agents` | `BaseAgent`, **`BaseSpecialist`**, `GenerativeLayer`, `Skill`, `SkillRegistry`, `AgentResponse`, `TraceStep`, `InputRequired`. `BaseSpecialist` is the extension point for a domain agent. |
| `erebots.agents.specialist` | The same `BaseSpecialist`, if you prefer the narrower import. |
| `erebots.agents.orchestrator` | `OrchestratorAgent` — routing, fan-out and composition. |
| `erebots.agents.a2a` | `A2AClient`, `create_a2a_app` — expose an agent over A2A, or talk to one. |
| `erebots.agents.mcp` | `MCPClient` — discover and invoke MCP tools; `MCPToolError`, raised when a tool runs and reports failure. |
| `erebots.agents.loader` | `load_agents`, `configured_urls` — read `agents.yaml` and discover the agents it lists through their A2A cards. |
| `erebots.server` | `create_app`, `create_chat_router`, `default_session_resolver`, `format_sse_event`, `sse_response`. |
| `erebots.runtime` | `AgentEvent`, `EventType`, `ConversationService` — the frontend contract — plus `render_user_context` and the `UserContextRenderer` alias, which decide what a turn tells the model about the user, and `UserContextStoreProtocol` / `UserContextLike`, which are what a replacement store is held to. An application swaps the renderer, the store or the claims mapper without subclassing anything. |
| `erebots.persistence` | `Base`, `make_engine`, `make_session_factory`, and `UserContextStore`, which an application constructs and injects. The framework's concern in that row is the conversation: creating it, and reading and writing the chat history. Everything else in it belongs to whichever agent owns that data, which reaches it through `merge_json_column` — a row-locked, column-wise merge, so two agents writing different fields of one user at the same time cannot overwrite each other. The `UserContext` it returns is deliberately **not** listed: code against `UserContextLike` in `erebots.runtime`, which is the shape, rather than against the dataclass, which is one implementation of it. |
| `erebots.config` | `settings`, the `Settings` type for your own annotations, and the typed environment readers `get_bool`, `get_int`, `get_list`, `get_str` for your own configuration module. |
| `erebots.security.keycloak` | OIDC helpers, for an application that wires its own auth routes. |
| `erebots.security.internal_token` | `InternalTokenProvider` — the machine-to-machine token an application injects into its A2A and MCP clients. |
| `erebots.security.claims` | `user_fields_from_claims` and the `ClaimsMapper` alias — how a token's claims become a new user's stored fields. The default is Keycloak-shaped; an application on another provider passes its own to `ConversationService`, which is why this is listed rather than internal. |

## Internal

Everything else, and one rule that makes this checkable rather than a matter of
taste: **a package with no `__init__.py` is internal.** That is true today of
`erebots/api/` and `erebots/api/routes/`. Any name starting with `_` is internal
wherever it lives.

`erebots/security/` is the documented exception to that rule. It has no package
`__init__.py`, but each of its modules is either imported by the example
application or named in the framework's own documentation as something to
replace, so all three are listed above and supported as module paths. The rule
holds for the directory; the three named modules are carved out of it
deliberately rather than by oversight. Anything added there later is internal
until it appears in the table.

The test of whether a path belongs in the Supported table is whether an
application outside `erebots/` has to import it. If the example app needs it,
so will anyone else's — publishing it is the honest move, and leaving it
unlisted only means nobody notices when it breaks.

`erebots.api.auth_middleware` in particular is **not** an extension point. It is
plumbing that `create_app` wires; its route tables and policy may change in any
release.

## Experimental

Four of the paths above are **carved out of the 1.0 promise**, and the carve-out
is named here rather than left implied. They exist, they will keep existing, and
they are safe to import — but their *shape* is still moving and may change in a
minor release.

A version number that quietly excepted things would be worth less than one that
says where its edges are. These are the edges:

- `erebots.agents.a2a.A2AClient` — the streaming and resume surface
- `erebots.agents.mcp.MCPClient` — error reporting is not final
- `erebots.agents.GenerativeLayer` — currently OpenAI-compatible only
- `erebots.runtime.ConversationService`

## What the policy is worth at 1.0

- A supported path **breaks only in a major release**, with the four exceptions
  named under [Experimental](#experimental) above. That is what the version number
  means, and it is the whole reason the table is short: every row is a commitment,
  so nothing is listed that does not need to be.
- A break is an entry in [`CHANGELOG.md`](../CHANGELOG.md), under `Changed` or
  `Removed`, naming the path and its replacement. A removal without a replacement
  named is a defect in the release, not a detail of it.
- A supported path may gain arguments with defaults, or new exports beside it, in
  a minor release. Existing calls keep working.
- Internal paths change without notice and without a changelog entry. What counts
  as internal is decided by the rule under [Internal](#internal) above, which is
  checkable rather than a matter of taste.
- `tests/test_public_api.py` asserts that every path in the Supported table
  imports, and pins `__version__` against `pyproject.toml`. If you rely on
  something that is not in that table, nothing protects you — not even at 1.0.

## Adding to the table

A path becomes supported when it has a reason to be used from outside, a test in
`tests/test_public_api.py`, and a row here. Prefer exporting from an existing
package over creating a new one.
