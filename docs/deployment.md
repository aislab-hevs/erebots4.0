# Deployment (full stack with Docker)

`docker compose up --build` brings up the whole backend: Traefik (TLS),
Keycloak (IdP), Postgres (Keycloak's DB), Postgres (app DB), and the API.
Your LLM stays on the host and is reached from the containers via
`host.docker.internal`.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Compose
- [mkcert](https://github.com/FiloSottile/mkcert) for local HTTPS
- An OpenAI-compatible LLM on the host (e.g. llama.cpp on `:8080`)

```bash
# local TLS certs for *.erebots.local
mkcert -install
mkdir -p traefik/certs && (cd traefik/certs && mkcert "*.erebots.local")

# /etc/hosts
#   127.0.0.1 app.erebots.local auth.erebots.local
```

## Configure

```bash
cp env.example .env
```

The file is read from the **working directory** the process is started in, so
keep it next to whatever you run (`app.py`, `demo.py`, or your own application).
Set `EREBOTS_ENV_FILE` to an absolute path to keep it elsewhere, or
`EREBOTS_IGNORE_DOTENV=1` to ignore it entirely and configure purely from the
environment. A real environment variable always wins over a value in the file.

Key settings in `.env`:

- LLM: `*_api_base_url=http://host.docker.internal:8080/v1` (the host's
  llama.cpp — see [running-a-model.md](running-a-model.md) to get one up).
- **`internal_client_secret`** — with `enable_auth=true` the app refuses to start
  on the placeholder value (a security guard). Set a real secret **and put the
  same value** on the `erebots-internal` client in `keycloak/erebots-realm.json`
  (it is imported at startup).
- `cors_origins` — include any separate frontend origin (e.g.
  `http://localhost:5173` for the web SPA).
- **`allow_loopback_tool_reads`** — `true` for this example, `false` in the
  framework. `app.py` mounts three Gradio tool servers, and Gradio answers an MCP
  call by calling itself back over loopback without an `Authorization` header, so
  its `GET`s need the exception. `demo.py` does not: its FastMCP tool servers use
  POST and send their own bearer. Enabling it trusts the reported peer address,
  so only set it where `/tools/*` is not routable from outside the deployment.

## Run

```bash
docker compose up --build
```

- App / Gradio UI: **https://app.erebots.local** (login `demo` / `demo`).
- Keycloak admin: **https://auth.erebots.local**.

The demo user and password are dev-only — remove them and rotate all secrets
before any real deployment.

## About the example realm

`keycloak/erebots-realm.json` is an **example** realm export: it ships with no
key material (no `KeyProvider` components), so Keycloak generates fresh RS256
signing keys the first time it imports the realm — every deployment gets its
own keys. Never reuse a realm export that contains `privateKey`/`secret`
entries: anyone holding those values can forge tokens your API would accept.
The `erebots-internal` client ships with the well-known placeholder secret,
which the app refuses at startup (see the `internal_client_secret` guard
above), so a deployer is forced to set their own.

## Web frontend (Vue)

**Served by the stack (recommended):** set `frontend=web` in `.env` and the API
serves the [Vue SPA](../examples/web/) at `https://app.erebots.local` (same
origin). Auth is the server-side Keycloak login: the browser is redirected to
Keycloak, the session cookie rides along on every `/api/*` call — no
keycloak-js involved. `frontend=gradio` serves the Gradio UI instead; both
consume the same orchestrator core.

**Cross-origin dev:** serve the SPA separately
(`cd examples/web && python3 -m http.server 5173`) against a local API. In
`index.html`, `apiBase` defaults to `http://127.0.0.1:7860` on port 5173; set
`useKeycloak: true` for browser login via keycloak-js (bearer). The
`erebots-webapp` client already allows `http://localhost:5173` in its
`redirectUris` and `webOrigins`, but **`cors_origins` does not include it** —
add it there yourself, or the API rejects the cross-origin request.

## Auth model

- **Users**: Keycloak OIDC. The browser flow sets an HttpOnly cookie; SPAs send a
  bearer. Both are validated the same way.
- **Machine-to-machine**: the internal `/a2a/*` and `/tools/*` channels require a
  Keycloak `client_credentials` token (`azp == erebots-internal`), minted by the
  app's `InternalTokenProvider`.

> **Trust boundary.** The internal service account is fully trusted: A2A calls
> carry the acting `user_id` in message metadata, so any holder of
> `internal_client_secret` can act as **any** user. Treat that secret like a
> root credential — inject it only via the environment, rotate it, and never
> commit it.

## Production checklist

- Rotate `internal_client_secret` and remove the demo realm user.
- Keep `erebots-webapp` on PKCE (`S256`, the bundled default) or make it
  confidential, and trim its `webOrigins` to the origins you actually serve —
  the bundled list includes the local development ones.
- Set `jwt_cookie_secure=true` (default) and serve over HTTPS.
- Set `a2a_persistent_tasks=true` if you run more than one API instance, so A2A
  tasks live in the database rather than in one process's memory.
