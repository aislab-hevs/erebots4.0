"""Template system for agent prompts.

Configures the Jinja2 environment for loading and rendering
system prompt templates. Templates are ``.j2`` files located
in the same directory as this module.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_PROMPTS_DIR = Path(__file__).parent

# Blocks are trimmed so the control flow in a template does not leave blank
# lines in the prompt the model actually reads.
_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_name: str, **kwargs) -> str:
    """Render a Jinja2 template for a system prompt.

    Loads the template from the filesystem and renders it with
    the provided variables.

    Args:
        template_name: Template file name (e.g., ``"orchestrator.j2"``).
        **kwargs: Variables to pass to the template for rendering.

    Returns:
        String with the rendered prompt.

    Raises:
        jinja2.TemplateNotFound: No such template in the prompts directory.
    """
    template = _env.get_template(template_name)
    return template.render(**kwargs)
