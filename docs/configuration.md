# Configuration

Every setting is an environment variable, read once at import into a frozen
`Settings` dataclass in [`erebots/config.py`](../erebots/config.py).

## Where values come from

In order, each winning over the next:

1. A real environment variable.
2. A `.env` file in the **working directory the process was started in** — not
   next to the package. Point `EREBOTS_ENV_FILE` at an absolute path to keep it
   elsewhere, or set `EREBOTS_IGNORE_DOTENV=1` to skip the file entirely and
   configure purely from the environment.
3. The default in the table below.

`env.example` is a complete, commented template: `cp env.example .env`.

The example application has its own settings for the domain it ships — the
personal and conversion agents — in
[`specialists/config.py`](../specialists/config.py). They are listed at the
bottom of this page. The framework never reads them.

---

## Framework settings

### Authentication and identity

| Setting | Default | What it does |
|---|---|---|
| `enable_auth` | `true` | Master switch. Off, every endpoint is open — development only. |
| `keycloak_url` | `http://localhost:8080` | Base URL used when no more specific one is set. |
| `keycloak_public_url` | `http://localhost:8080` | The URL a *browser* is redirected to. Differs from the internal one behind a proxy. |
| `keycloak_internal_url` | `http://localhost:8080` | The URL the *server* reaches Keycloak on (container hostname in Docker). |
| `keycloak_realm` | `erebots` | Realm name. |
| `keycloak_client_id` | `erebots-webapp` | The public client the browser logs in with. |
| `keycloak_client_secret` | unset | Only if that client is made confidential. |
| `app_callback_url` | `http://localhost:7860/api/auth/callback` | OIDC redirect target; must be registered on the client. |
| `jwt_cookie_name` | `access_token` | Cookie holding the access token. |
| `id_token_cookie_name` | `id_token` | Cookie holding the id token, needed for logout. |
| `jwt_cookie_secure` | `true` | HTTPS-only cookies. Turn off only for plain-HTTP local development. |
| `internal_client_id` | `erebots-internal` | Service account for the internal channels; checked as the `azp` claim. |
| `internal_client_secret` | *(empty)* | **Root-equivalent.** With `enable_auth=true` the app refuses to start on an empty or placeholder value. |

### Server and frontend

| Setting | Default | What it does |
|---|---|---|
| `app_host` | `0.0.0.0` | Bind address. |
| `app_port` | `7860` | Bind port; also how an agent builds its own tool-server URLs. |
| `frontend` | `gradio` | Which UI the example app serves — `gradio` or `web`. |
| `ui_path` | `/` | Where that UI is mounted. |
| `brand_name` | `EREBOTS` | Name shown in the bundled UIs. |
| `cors_origins` | `["http://localhost", "http://localhost:7860"]` | Origins allowed to call the API. A separately served SPA must be added here. |
| `database_url` | `postgresql+asyncpg://erebots:erebots@erebots-app-db:5432/erebots` | SQLAlchemy URL. Use `sqlite+aiosqlite:///./dev.db` locally. |
| `agents_config_path` | `agents.yaml` | Agent list, resolved from the working directory. |

### The orchestrator's own model

The orchestrator routes with an LLM and answers directly when no specialist
matches, so it has a model of its own.

| Setting | Default | What it does |
|---|---|---|
| `orchestrator_api_base_url` | `http://localhost:8080/v1` | Any OpenAI-compatible endpoint. |
| `orchestrator_api_key` | `EMPTY` | Most local servers ignore it; a hosted API does not. |
| `orchestrator_model` | `local` | Model id. |
| `orchestrator_thinking` | `false` | Ask the model for explicit reasoning. |

### Routing and composition

The knobs that decide how a turn is distributed. These change answers, not just
performance.

| Setting | Default | What it does |
|---|---|---|
| `orchestrator_max_specialists` | `3` | Upper bound on how many specialists one turn may fan out to. `1` disables fan-out. |
| `multi_agent_composition` | `synthesis` | How several answers become one: `synthesis` rewrites them into a single reply, `sections` keeps them labelled. |
| `routing_include_skills` | `true` | Put each agent's skills in the routing prompt, not just its description. Turn off for a small routing model that cannot handle the longer prompt. |
| `tool_loop_max_iterations` | `3` | How many times a specialist may call tools before it must answer. Bounds a model that keeps asking for tools. |
| `discovery_retry_seconds` | `30` | Throttle for retrying an MCP server that failed discovery. |
| `multimodal_enabled` | `true` | Accept images in a turn. |
| `enable_parser_retry` | `false` | Re-prompt once when the router's reply cannot be parsed. |

### Conversation limits

Bounds on what one request may carry. They exist because the chat body is where
untrusted input becomes prompt material.

| Setting | Default | What it does |
|---|---|---|
| `chat_history_max_messages` | `20` | Most recent turns kept in the prompt. |
| `chat_request_max_bytes` | `8388608` (8 MiB) | Whole request body. |
| `chat_message_max_chars` | `20000` | One message's text. |
| `chat_images_max_per_message` | `4` | Images per message. |
| `chat_image_max_bytes` | `5242880` (5 MiB) | One image. |

### A2A

| Setting | Default | What it does |
|---|---|---|
| `a2a_persistent_tasks` | `false` | Keep A2A tasks in the database instead of memory. **Required when more than one API instance serves the same users.** |
| `a2a_parked_task_ttl` | `900` (15 min) | How long a task parked on a question stays resumable. |
| `a2a_conformance_agent` | `false` | Mount the `/a2a/tck` and `/a2a/tck-minimal` fixtures. For a conformance run only — they serve canned payloads. See [tck.md](tck.md). |

### Relaxations — off by default

Both widen the attack surface. `SECURITY.md` lists them as accepted risks.

| Setting | Default | What it relaxes |
|---|---|---|
| `allow_loopback_tool_reads` | `false` | Serves unauthenticated `GET` under `/tools/*` to a loopback peer. Needed by the example app's Gradio tool servers, which call themselves back without an `Authorization` header. It trusts the reported peer address, so anything that can make a request look local can read those endpoints. Retired, not re-scoped, when those mounts move to FastMCP. |
| `a2a_push_allow_insecure_webhooks` | `false` | Accepts push-notification webhooks on plain HTTP or non-public addresses. The default policy is what stops a caller aiming this server at its own network. For the conformance suite and local development. |

---

## Example-domain settings

Not part of the framework. Defined in
[`specialists/config.py`](../specialists/config.py) and read at bootstrap rather
than at import.

Each specialist takes the same four: `<agent>_api_base_url`, `<agent>_api_key`,
`<agent>_model` and `<agent>_thinking`, for `conversion` and `personal`.
Defaults are `http://localhost:8080/v1`, `EMPTY`, `local` and `false`.

Adding a specialist adds its own block there and touches nothing under
`erebots/`.
