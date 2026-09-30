"""Personal-agent skill descriptors mirroring the MCP tools.

Metadata only: this is what the A2A card advertises, while execution happens on
the MCP server (see :mod:`specialists.personal.tools`). The argument schemas live
there, on the tools themselves, where a caller can read them — the A2A card has
no field for one.

``user_id`` is not advertised on purpose. The agent pins it server-side from the
trusted request context (:meth:`PersonalAgent.prepare_tool_args`), so it is never
something a caller supplies.
"""

from __future__ import annotations

from erebots.agents.skill import Skill


def get_personal_skills() -> list[Skill]:
    """Return the descriptors of the personal MCP tools."""
    return [
        Skill(
            name="update_profile",
            description=(
                "Update fields in the user's profile (name, age, weight, "
                "height, dietary preferences, allergies, ...). Provide "
                "only the fields to change."
            ),
        ),
        Skill(
            name="manage_consent",
            description=(
                "Update privacy or consent flags for the user (analytics, "
                "marketing, ...). Provide only the consents to change."
            ),
        ),
        Skill(
            name="get_user_context",
            description="Retrieve the user's stored profile, preferences and consents.",
        ),
    ]
