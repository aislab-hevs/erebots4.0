"""Prompt construction for the personal agent.

Loads and renders the personal agent's Jinja2 system prompt from the
template that sits next to this file, the way every specialist keeps its own.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_PROMPTS_DIR = Path(__file__).parent
_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_personal_prompt() -> str:
    """Render the personal agent system prompt."""
    return _env.get_template("personal.j2").render()


# Module-level constant for back-compat with previous imports.
PERSONAL_SYSTEM_PROMPT = build_personal_prompt()
