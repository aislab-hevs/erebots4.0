"""Prompt construction for the conversion agent.

Renders the Jinja2 system prompt from the template next to this file
(mirrors ``specialists.personal.prompts``).
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


def build_conversion_prompt() -> str:
    """Render the conversion agent system prompt."""
    return _env.get_template("conversion.j2").render()


CONVERSION_SYSTEM_PROMPT = build_conversion_prompt()
