"""Conversion agent: unit conversion over a deterministic MCP tool.

The counterpart to the generative specialists — its answers come from
arithmetic, not from a model's arithmetic. The tool-calling loop and MCP
discovery live in :class:`erebots.agents.specialist.BaseSpecialist`; this
module contributes the prompt, description and skills.
"""

from __future__ import annotations

from erebots.agents.specialist import BaseSpecialist
from specialists.conversion.prompts import CONVERSION_SYSTEM_PROMPT
from specialists.conversion.skills import get_conversion_skills


class ConversionAgent(BaseSpecialist):
    """Specialist that converts quantities between units of measure."""

    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        model: str,
        mcp_server_urls: list[str] | None = None,
        thinking: bool = False,
    ):
        super().__init__(
            agent_name="conversion",
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            system_prompt=CONVERSION_SYSTEM_PROMPT,
            description=(
                "Converts quantities between units of measure: mass "
                "(kg, g, lb, oz), length (m, cm, in, ft, mi), energy "
                "(kcal, kJ) and volume (l, dl, ml). Route here when the "
                "request is to express a value in a different unit."
            ),
            mcp_server_urls=mcp_server_urls or [],
            thinking=thinking,
        )
        self.setup_skills()

    def setup_skills(self):
        """Advertise the conversion MCP tools as A2A skills."""
        for skill in get_conversion_skills():
            self.skills.register(skill)
