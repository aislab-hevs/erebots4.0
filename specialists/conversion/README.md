# The conversion agent — reference implementation

This is the smallest complete specialist in the repository: **255 lines across
six files**, of which 133 are the tool itself. It is the one to read first, and
the one to copy.

It converts quantities between units. That sounds trivial, and it is the point:
an LLM asked to convert 70 kg to pounds will answer, confidently, and be wrong
often enough to matter. Here the model decides *that* a conversion is needed and
*which* one; a deterministic MCP tool decides what the answer is. The whole
protocol path — routing, A2A, MCP, the tool loop — is exercised by a capability
whose correctness you can check by hand.

## The six files

| File | Lines | What it contributes |
|---|---|---|
| [`agent.py`](agent.py) | 47 | The `BaseSpecialist` subclass. Prompt, description, skills — nothing else. |
| [`tools.py`](tools.py) | 133 | The MCP server and the arithmetic. The only file with real logic. |
| [`skills.py`](skills.py) | 25 | What the A2A card advertises. Metadata only. |
| [`prompts.py`](prompts.py) | 25 | Renders the Jinja2 template below. |
| [`conversion.j2`](conversion.j2) | 17 | The system prompt. |
| [`__init__.py`](__init__.py) | 8 | Exports `ConversionAgent`, `build_conversion_mcp`, `convert`. |

Notice what is *not* here: no tool-calling loop, no MCP discovery, no A2A
server, no streaming, no task handling. All of that is in
`erebots.agents.specialist.BaseSpecialist` and the framework around it. A
specialist contributes a prompt, a description and a list of skills.

## Four decisions worth copying

**The description is routing input.** The orchestrator picks a specialist by
reading it, so it names the dimensions covered and ends with an explicit trigger
— *"Route here when the request is to express a value in a different unit."*
A vague description produces vague routing.

**The prompt forbids the model from doing the work.** `conversion.j2` says
"Always call the tool. Never convert in your head and never state a factor from
memory: the tool is the authority, and a number you produce yourself is exactly
the error this agent exists to prevent." Without that, a model will shortcut the
tool on easy cases — which are precisely the ones where a wrong answer looks
right.

**The prompt also says what the agent is not.** Several specialists can answer
one turn and their answers are merged before the user sees them, so a handoff
sentence ("for that, please ask the other specialist...") only confuses. The last
paragraph tells the agent to answer its part and stop.

**The arithmetic stays callable without MCP.** `convert()` is a plain function;
`convert_units` is a thin wrapper that MCP exposes. That is what lets the logic
be tested with no server, no transport and no event loop.

## Two refusals

Both are in `tools.py`, and both are the difference between a tool and a
plausible-answer generator:

- **`nan` and `inf` are rejected.** The model fills these arguments in, so they
  are reachable through a JSON literal or a stray division. They would convert
  happily and put "inf kg = inf lb" in front of the user as if it were a result.
- **`cal` is not mapped to `kcal`.** In nutrition it colloquially means kcal; it
  strictly means a thousandth of one. A deterministic tool must not resolve that
  ambiguity silently, so it refuses and names what it does support.

Cooking measures (cup, tablespoon) are absent for the same reason: their size
depends on the convention.

## Building your own

1. Subclass `BaseSpecialist` the way `agent.py` does — see
   [docs/adding-an-agent.md](../../docs/adding-an-agent.md).
2. Put the deterministic part behind an MCP tool server — see
   [docs/adding-a-tool.md](../../docs/adding-a-tool.md).
3. Declare it in [`specialists/registry.py`](../registry.py) and list its
   `/a2a/<name>` URL in `agents.yaml`.

`demo.py` at the repository root runs this agent and nothing else, without
Gradio. It is the shortest way to see the whole path run: routing, A2A, an MCP
tool call and persistence.
