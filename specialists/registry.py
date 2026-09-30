"""The example app's in-process specialist agents.

This is the single place to declare local specialists: add or remove an entry
in :func:`build_specialists` and the composition root (``bootstrap.py``) wires
the rest generically — mounting, A2A apps and token injection all iterate over
the returned dict. Adding an agent never touches the infrastructure (orchestrator,
persistence, auth) in ``bootstrap.py``.

To route to a *remote* agent instead, add its URL to ``agents.yaml`` — no code
here. For a local agent, add it here *and* list its ``/a2a/<name>`` URL in
``agents.yaml`` so the orchestrator discovers it (the agent name below must match
that URL's last segment).
"""

from __future__ import annotations

from erebots.agents.specialist import BaseSpecialist
from erebots.config import Settings
from specialists.config import DomainSettings
from specialists.conversion import ConversionAgent
from specialists.personal import PersonalAgent


def build_specialists(
    settings: Settings, domain_settings: DomainSettings | None = None
) -> dict[str, BaseSpecialist]:
    """Build the in-process specialist agents, keyed by name.

    The key is the agent's name: it becomes the A2A mount path (``/a2a/<name>``)
    and must match the URL listed for it in ``agents.yaml``. ``domain_settings``
    defaults to the environment as it is now.
    """
    if domain_settings is None:
        domain_settings = DomainSettings.from_env()

    def mcp(tool: str) -> str:
        """Endpoint of a tool server built with the MCP SDK."""
        return f"http://localhost:{settings.app_port}/tools/{tool}/mcp"

    return {
        "conversion": ConversionAgent(
            api_key=domain_settings.conversion_api_key,
            api_base_url=domain_settings.conversion_api_base_url,
            model=domain_settings.conversion_model,
            mcp_server_urls=[mcp("conversion")],
            thinking=domain_settings.conversion_thinking,
        ),
        "personal": PersonalAgent(
            api_key=domain_settings.personal_api_key,
            api_base_url=domain_settings.personal_api_base_url,
            model=domain_settings.personal_model,
            mcp_server_urls=[mcp("personal")],
            thinking=domain_settings.personal_thinking,
        ),
    }
