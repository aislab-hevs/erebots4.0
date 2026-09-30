# Erebots4.0 — web frontend example

A single-file, no-build web client (Vue 3 + `keycloak-js` from a CDN) that
streams a chat from the framework's canonical SSE endpoint
(`POST /api/chat/stream`). It's the *third* frontend on the same headless core
(alongside the Gradio UI and the CLI) — proof that the core is frontend-agnostic.
The same pattern works unchanged in React.

## Why `fetch`, not `EventSource`

`EventSource` only does GET. The chat endpoint is a POST (it carries the message
body), so the example reads the SSE stream with `fetch()` + a stream reader and
parses the `event:` / `data:` frames by hand — a few lines in `index.html`.

## Three auth modes, picked from the config

The `CONFIG` block at the top of `index.html` decides which one you get:

| Mode | When | How the request is authenticated |
|---|---|---|
| `cookie` | `apiBase: ""` — the API serves this page | The Keycloak session cookie the server set at login. No `keycloak-js` involved. |
| `none` | cross-origin, `useKeycloak: false` **(the shipped default)** | Nothing. Works against an API started with `enable_auth=false`. |
| `keycloak` | cross-origin, `useKeycloak: true` | Browser login via `keycloak-js` (public client, PKCE `S256`) → a bearer token in `Authorization`. |

The file ships with `useKeycloak: false`, so serving it from the dev static
server talks to an unauthenticated API. Set it to `true` for the Keycloak flow.

## Run it

1. **Serve the file** from any static host, e.g.:
   ```bash
   cd examples/web && python -m http.server 5173
   ```
   `apiBase` already defaults to `http://127.0.0.1:7860` on port 5173.
2. **CORS** — add `http://localhost:5173` to `cors_origins` in your `.env`. It is
   *not* in the default, so the cross-origin request fails without this step.
3. **Keycloak**, only for `useKeycloak: true` — the bundled realm already lists
   `http://localhost:5173/*` in `redirectUris` and `http://localhost:5173` in
   `webOrigins` for `erebots-webapp`, so there is nothing to add. Note that the
   same origin is missing from `post.logout.redirect.uris`, so a logout from the
   dev SPA is rejected; add it there if you need it.

Open the page → chat streams token-by-token over SSE.

## What it demonstrates

- **Auth**: all three ways the API accepts an identity — session cookie, Keycloak
  bearer, or none at all — behind one `extract_user_token` on the server side.
- **Streaming**: the neutral `AgentEvent` stream (`routing` / `token` / `final`
  / `error`) rendered in the browser — identical to what the CLI prints.

## Third-party assets

The page loads three things from public CDNs at runtime; none is copied into this
repository, so nothing here redistributes them. They are named because a reader
is entitled to know what the page pulls in before opening it.

| | Licence | |
|---|---|---|
| Vue 3 | MIT | <https://vuejs.org> |
| keycloak-js 25 | Apache-2.0 | <https://www.keycloak.org> |
| Zalando Sans Expanded | SIL OFL 1.1 | Google Fonts |

All three are fetched, never copied into the repository — see [NOTICE](../../NOTICE).
