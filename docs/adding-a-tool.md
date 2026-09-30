# Adding a tool

Agents talk to each other over A2A. They reach **capabilities** over MCP, and a
capability is its own server: the agent discovers what it offers at runtime and
calls it by name.

That separation is the point. A specialist's prompt decides *when* to convert a
unit; the tool decides *what* the answer is. Putting the arithmetic in the model
would make it a guess, and a guess that is right most of the time is the worst
kind. Anything that must be deterministic — a calculation, a database read, a
call to a system of record — belongs on this side of the line.

The worked example throughout is
[`specialists/conversion/tools.py`](../specialists/conversion/tools.py), the
whole of which is one tool.

## 1. Build the server

```python
from mcp.server.fastmcp import FastMCP


def build_conversion_mcp() -> FastMCP:
    server = FastMCP("conversion", stateless_http=True)

    @server.tool()
    async def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert a quantity between units of the same kind.

        Supports mass (kg, g, mg, lb, oz), length (m, cm, mm, km, in, ft, mi),
        energy (kcal, kJ) and volume (l, dl, ml). Use it for every conversion
        instead of computing one yourself.

        Args:
            value: The amount to convert.
            from_unit: Unit the value is expressed in (e.g. "kg", "pounds").
            to_unit: Unit to convert it to (e.g. "lb", "kilograms").

        Returns:
            The converted quantity, e.g. ``70 kg = 154.324 lb``.
        """
        result, source, target = convert(value, from_unit, to_unit)
        return f"{value:.6g} {source} = {result:.6g} {target}"

    return server
```

Three things in that snippet are load-bearing.

**The docstring is the tool's interface, not decoration.** MCP publishes it, and
the framework hands it to the model as the tool description — it is the only
thing the LLM knows about your tool. Write it for that reader: say what the tool
does, what it covers, and when to reach for it. The last sentence above ("Use it
for every conversion instead of computing one yourself") exists because without
it a model will happily do the arithmetic itself.

**The type annotations become the tool's schema.** `value: float` is what stops
the model sending `"seventy"`. There is no separate schema to declare — and
nowhere to declare one, which is why `Skill` carries no input schema either.

**`stateless_http=True` is required.** The framework mounts the server's
Streamable HTTP app and calls it one request at a time; a stateful server would
issue a session id and expect the client to hold it open.

### Keep the logic callable without MCP

`convert_units` above is a thin wrapper over a plain `convert()` function. That
is deliberate: the arithmetic can then be tested directly, with no server, no
transport and no event loop. Put the decisions in a normal function and let the
tool be the adapter.

### Failing is part of the contract

Raise an ordinary exception with a message aimed at the model:

```python
raise ValueError(f"Cannot convert {source} ({source_dim}) to {target} ({target_dim}).")
```

The specialist's tool loop turns that into an observation and lets the model try
again or explain itself, rather than ending the turn. A tool that returns a
plausible-looking wrong answer instead of raising is the failure mode to avoid —
`convert()` rejects `nan` and `inf` for exactly this reason.

Calling `MCPClient` yourself rather than through a specialist, you get
`MCPToolError` — the tool ran and said no, which is a different thing from the
server being unreachable, and worth distinguishing before you decide to retry.
Its message is the server's own, so it can be handed to a model unchanged.

## 2. Mount it

Tool servers are passed to the framework's app factory, which mounts each one at
`/tools/<name>` with its endpoint at `/tools/<name>/mcp`:

```python
from erebots.server import create_app

app = create_app(
    orchestrator,
    mcp_apps={"conversion": build_conversion_mcp()},
)
```

In the example application this happens in
[`bootstrap.py`](../bootstrap.py), which returns the servers alongside the
specialists, and [`app.py`](../app.py) forwards them.

## 3. Point an agent at it

A specialist takes the endpoints it should discover tools from:

```python
ConversionAgent(
    api_key=...,
    api_base_url=...,
    model=...,
    mcp_server_urls=[f"http://localhost:{settings.app_port}/tools/conversion/mcp"],
)
```

[`specialists/registry.py`](../specialists/registry.py) has an `mcp()` helper
that builds that URL. Discovery happens on first use rather than at startup, and
a server that does not answer is skipped and retried on a later turn (throttled
by `discovery_retry_seconds`) — so a tool server that is slow to come up neither
blocks the application nor stays lost until a restart.

Several agents may point at the same server, and one agent may point at several.
Nothing links a tool server to the agent that uses it except this URL — which is
why a tool server can equally well run in another container, in another language,
or on another host.

## Authentication

`/tools/*` is an internal channel. With `enable_auth=true` every request needs
the service-account bearer, checked by the framework's auth dispatcher against
the `azp` claim; `bootstrap.py` gives each specialist's MCP client a token
provider so its calls carry one. Your tool server does not implement any of this
and should not try to.

A tool server is therefore **not** an authorization boundary. If the model must
not choose a value — a user id, a tenant — pin it server-side on the agent by
overriding `prepare_tool_args`, as the personal agent does with `user_id`. The
tool sees only what the agent sends it, and
`tests/test_specialist_stream.py` holds the framework to applying that hook on
both the streaming and the non-streaming tool loop.

## A note on Gradio-hosted tool servers

A tool server can also be hosted by Gradio rather than built with the MCP SDK.
That is the older path, and it is the reason the `allow_loopback_tool_reads`
setting exists: Gradio answers an MCP call by calling itself back over loopback
with no `Authorization` header, so those `GET`s need an exception the
framework's own mounts do not. `SECURITY.md` lists it as an accepted risk, off
by default. Nothing in this repository mounts one, so nothing here needs the
setting; build new tool servers the way this page describes.

Do not copy that pattern. Build new tools the way this page describes.

## See also

- [adding-an-agent.md](adding-an-agent.md) — the agent that calls your tool.
- [`specialists/conversion/`](../specialists/conversion/) — the reference
  implementation, end to end, with its own
  [README](../specialists/conversion/README.md).
