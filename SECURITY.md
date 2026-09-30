# Security policy

## Supported versions

`erebots` is **1.0.x**. Fixes land on the development branch and are released
from there; there are no maintained older lines, so a report against anything
before 1.0.0 will be answered with an upgrade.

## Reporting a vulnerability

Report privately to **elia.pacioni@hevs.ch**. Please do not open a public issue
for a suspected vulnerability.

Useful in a report: what an attacker gains, the affected version or commit, and
the smallest reproduction you have. A configuration description is enough — you
do not need a working exploit.

What to expect:

- an acknowledgement within **10 working days**;
- an assessment, and a fix or a documented decision not to fix;
- credit in the changelog entry if you want it.

There is no bounty, and we do not run a CVE process.

## Scope

**In scope** — the framework in `erebots/`:

- the authentication dispatcher and the user and machine-to-machine token paths
  (`erebots/api/`, `erebots/security/`);
- the HTTP surface `create_app` builds: the SSE chat endpoint, the A2A mounts and
  the MCP tool mounts;
- A2A task handling, including resume ownership and parked-task expiry;
- push-notification webhook validation (SSRF, DNS rebinding).

**Out of scope:**

- the example domain in `specialists/` and the example applications `app.py` and
  `demo.py` — they exist to demonstrate the framework, not to be deployed;
- `keycloak/erebots-realm.json`. It is a development fixture whose credentials
  are documented placeholders, and `bootstrap.py` refuses to start with them when
  authentication is enabled. It declares no key providers, so Keycloak generates
  a fresh set on import — no signing key is shipped with this repository;
- findings that require an attacker to already control the host or the
  configuration.

## Known accepted risks

Both are off by default. Turning either on is a deliberate deployment choice, and
each is documented where it is defined in `erebots/config.py`.

| Setting | What it relaxes |
|---|---|
| `allow_loopback_tool_reads` | Serves unauthenticated `GET` under `/tools/*` from a loopback peer. It trusts the reported client address, so anything that can make a request look local can read those endpoints. Needed only for a deployment that mounts a re-entrant UI there, such as a Gradio-hosted tool server. Nothing in this repository does, so nothing here needs it on. |
| `a2a_push_allow_insecure_webhooks` | Accepts push-notification webhooks on plain HTTP or non-public addresses. The default policy is what stops a caller aiming this server at its own network. Intended for the conformance suite and local development. |
