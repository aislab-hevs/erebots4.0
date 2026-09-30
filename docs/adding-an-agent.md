# Adding an agent

Erebots4.0 routes to agents over **A2A** (each agent is a URL). There are two cases:

1. An **autonomous agent** running elsewhere (another process/container/server,
   any language) → you only register its URL.
2. An agent **hosted in this app** → you write it, build it, and register it.

Neither case requires changing the framework (`erebots/`). The import-boundary
test in `tests/test_import_boundaries.py` enforces that.

## 1. Route to an autonomous agent (just a URL)

If an A2A v1.0 agent is already serving its card at, say,
`https://legal.example.com/a2a/legal`, add it to `agents.yaml`:

```yaml
agents:
  - http://localhost:7860/a2a/conversion  # hosted here
  - http://localhost:7860/a2a/personal     # hosted here
  - https://legal.example.com/a2a/legal   # autonomous, elsewhere
```

The orchestrator fetches its agent card (name / description / skills) and starts
routing to it. Done — nothing else.

## 2. Host a new specialist in this app

> The shortest complete example is
> [`specialists/conversion/`](../specialists/conversion/) — 255 lines across six
> files, with a [README](../specialists/conversion/README.md) walking through
> each. Read it alongside this page.

### a. Write the specialist

Subclass `BaseSpecialist` in your domain code (e.g. `specialists/weather/agent.py`).
You provide the prompt, description and skills; the base class handles the
tool-calling loop and MCP discovery.

Both names are also available from `erebots.agents`. See
[api-stability.md](api-stability.md) for what the framework supports importing.

```python
from erebots.agents import BaseSpecialist, Skill


class WeatherAgent(BaseSpecialist):
    def __init__(self, api_key, api_base_url, model, mcp_server_urls=None, thinking=False):
        super().__init__(
            agent_name="weather",
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            system_prompt="You are a weather assistant. Be concise.",
            description="Weather forecasts and current conditions.",
            mcp_server_urls=mcp_server_urls or [],   # MCP tool endpoints, if any
            thinking=thinking,
        )
        self.setup_skills()

    def setup_skills(self):
        # Skills are metadata advertised on the A2A card; execution flows
        # through the MCP tools discovered from mcp_server_urls.
        self.skills.register(Skill(name="forecast", description="Forecast for a city"))
```

To enforce a server-side value the LLM must not control (like the personal
agent pinning `user_id`), override `prepare_tool_args(self, tool_args, context)`.

### b. Build it

Add an entry to `build_specialists` in
[`specialists/registry.py`](../specialists/registry.py) — the single place where
local specialists are declared:

```python
from specialists.weather import WeatherAgent

def build_specialists(settings, domain_settings=None):
    ...
    return {
        "conversion": ConversionAgent(...),
        "weather": WeatherAgent(
            api_key=domain_settings.weather_api_key,
            api_base_url=domain_settings.weather_api_base_url,
            model=domain_settings.weather_model,
            mcp_server_urls=[mcp("weather")],   # if it has MCP tools at /tools/weather
        ),
    }
```

**The key is the agent's name**, and it does double duty: it becomes the A2A
mount path `/a2a/<name>` and it has to match the URL you list in `agents.yaml`
in step c.

Nothing else needs editing. `bootstrap.py` iterates over whatever this function
returns — building the A2A apps, injecting the service-account token — and the
mount loop itself is in the framework, at
[`erebots/server/app.py`](../erebots/server/app.py) (`for name, a2a_app in
a2a_apps.items()`). Neither file knows your agent exists.

If your agent brings its **own** deterministic capabilities, those are a separate
MCP server: see [adding-a-tool.md](adding-a-tool.md).

### c. Route to it

Add its local URL to `agents.yaml`:

```yaml
agents:
  - http://localhost:7860/a2a/conversion
  - http://localhost:7860/a2a/personal
  - http://localhost:7860/a2a/weather
```

That's it. The orchestrator now routes weather questions to your specialist.

## What you did *not* touch

- `erebots/` — the framework. Your agent subclasses `BaseSpecialist` and is
  resolved as a URL; the core never imports your domain.
- `bootstrap.py` — the composition root. It wires the orchestrator, persistence
  and auth, and iterates over `build_specialists` without naming any agent.
- The framework's A2A mount loop, the orchestrator and the loader — all generic.

Run `uv run pytest tests/test_import_boundaries.py` to confirm the boundary
still holds.
