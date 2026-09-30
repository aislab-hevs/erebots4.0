# Erebots4.0 documentation

Erebots4.0 is a protocol-first framework for multi-agent systems: a generalist
orchestrator routes each request to a domain specialist, and agents are decoupled
by **A2A** and **MCP** rather than by Python imports.

Start with the [project README](../README.md) for what it is and why. This page
is the map of everything else.

## Understand it

| Page | |
|---|---|
| [architecture.md](architecture.md) | The layers, the two protocol seams, one turn end to end, the event contract, and where state lives. **Read this first.** |
| [api-stability.md](api-stability.md) | Which import paths are supported, which are internal, and what the 1.0 promise is worth. |

## Run it

| Page | |
|---|---|
| [running-a-model.md](running-a-model.md) | Get an LLM up with llama.cpp — on a workstation with a GPU, or on a laptop. Nothing works without one. |
| [configuration.md](configuration.md) | Every setting, grouped by concern, with defaults and the reasoning where it is not obvious. |
| [deployment.md](deployment.md) | The full stack with Docker: Keycloak, Postgres, Traefik, TLS. |

## Extend it

| Page | |
|---|---|
| [adding-an-agent.md](adding-an-agent.md) | Add a specialist — hosted here, or autonomous on another host, where it is only a URL. |
| [adding-a-tool.md](adding-a-tool.md) | Add a deterministic capability as an MCP tool server. |
| [`specialists/conversion/`](../specialists/conversion/README.md) | The reference implementation: a complete specialist in 255 lines, answering from arithmetic rather than from a model's arithmetic. |

## Verify it

| Page | |
|---|---|
| [tck.md](tck.md) | Run the official A2A TCK against a live endpoint, and the record to fill in for a cited run. |
| [a2a-conformance.md](a2a-conformance.md) | What the agent cards declare, and how to read a TCK result without misquoting it. |

## Contribute

| Page | |
|---|---|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Setup, the two rules that are not obvious, commit and pull-request conventions. |
| [SECURITY.md](../SECURITY.md) | How to report a vulnerability, what is in scope, and the accepted risks. |
| [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Contributor Covenant 2.1. |
| [CHANGELOG.md](../CHANGELOG.md) | Notable changes, from the first public release onward. |
