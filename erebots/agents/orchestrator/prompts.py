"""Prompt construction for the orchestrator agent.

Provides functions to generate the system prompts used by the
orchestrator: the routing prompt (to determine the target agent)
and the main prompt (for fallback responses).
"""

from erebots.agents.prompts import render


def build_routing_prompt(agent_descriptions: str, max_specialists: int = 1) -> str:
    """Build the system prompt for request routing.

    The generated prompt instructs the model to name the specialists a request
    needs, based on the descriptions of available agents.

    Args:
        agent_descriptions: Formatted descriptions of registered agents,
            one per line in the format ``- name: description``.
        max_specialists: How many agents the router may name for one request.
            ``1`` keeps the single-target behaviour.

    Returns:
        System prompt rendered from the Jinja2 template.
    """
    return render(
        "orchestrator_router.j2",
        agent_descriptions=agent_descriptions,
        max_specialists=max_specialists,
    )


def build_synthesis_prompt() -> str:
    """Build the system prompt that merges several specialist answers into one.

    Returns:
        System prompt rendered from the Jinja2 template.
    """
    return render("synthesis.j2")


def build_orchestrator_prompt(agent_descriptions: str) -> str:
    """Build the main system prompt for the orchestrator.

    The generated prompt is used when the orchestrator responds
    directly (fallback), without delegating to a specialized agent.

    Args:
        agent_descriptions: Formatted descriptions of registered agents,
            one per line in the format ``- name: description``.

    Returns:
        System prompt rendered from the Jinja2 template.
    """
    return render("orchestrator.j2", agent_descriptions=agent_descriptions)
