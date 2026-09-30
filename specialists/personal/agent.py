"""Personal agent for user context management.

Specialist exposed over A2A; manages user profile, preferences and privacy
consents through MCP tools. The MCP tools require a ``user_id`` argument: the
agent **always** overrides it server-side with the trusted value from the
request context, so the LLM cannot write to a different user's record.

The tool-calling loop and MCP discovery live in
:class:`erebots.agents.specialist.BaseSpecialist`; this module contributes the
domain prompt, skills, and the ``user_id`` enforcement hook.
"""

from __future__ import annotations

from erebots.agents.specialist import BaseSpecialist
from specialists.personal.prompts import PERSONAL_SYSTEM_PROMPT
from specialists.personal.skills import get_personal_skills


class PersonalAgent(BaseSpecialist):
    """Specialist that owns user profile and consents via MCP tools."""

    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        model: str,
        mcp_server_urls: list[str] | None = None,
        thinking: bool = False,
    ):
        urls = mcp_server_urls or []
        super().__init__(
            agent_name="personal",
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            system_prompt=PERSONAL_SYSTEM_PROMPT,
            description=(
                "Handles anything about the user themselves: their name, "
                "personal details, preferences, dietary restrictions, "
                "health/lifestyle info they share, profile updates, and "
                "privacy/consent settings. Route here when the user is "
                "telling the system something about themselves, asking "
                "what is remembered about them, or updating their profile."
            ),
            mcp_server_urls=urls,
            thinking=thinking,
        )
        self.setup_skills()

    def setup_skills(self):
        """Advertise the personal MCP tools as A2A skills."""
        for skill in get_personal_skills():
            self.skills.register(skill)

    def prepare_tool_args(self, tool_args: dict, context: dict | None) -> dict:
        """Pin ``user_id`` to the trusted request identity.

        Server-side enforcement: the LLM never controls which user record is
        touched, even if it puts a different ``user_id`` in the tool call.
        """
        tool_args["user_id"] = (context or {}).get("user_id", "anonymous")
        return tool_args
