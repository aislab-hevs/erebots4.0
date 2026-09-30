"""Agent skill metadata and registry.

A ``Skill`` is pure metadata advertised on the agent's A2A card; the actual
execution flows through MCP tools discovered at runtime, not through the skill.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Skill:
    """A skill advertised by an agent (metadata only).

    Deliberately not a place for an input schema. A2A's ``AgentSkill`` has no
    field for one — it carries ``id``, ``name``, ``description``, ``tags``,
    ``examples``, the two mode lists and security requirements — so a JSON Schema
    set here could never reach the wire. The callable contract lives on the MCP
    tool that actually executes, where a client can read and validate it.

    Attributes:
        name: Unique name of the skill.
        description: Text description of the offered functionality.
    """

    name: str
    description: str


class SkillRegistry:
    """Registry of an agent's skills, keyed by name."""

    def __init__(self):
        """Start with an empty registry."""
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """Register (or overwrite) a skill by name."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Return a skill by name, or ``None``."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def to_a2a_skills(self) -> list[dict]:
        """Convert the skills to the A2A agent-card format.

        Emits exactly the four keys the card consumes. ``_build_card`` in
        :mod:`erebots.agents.a2a.server` reads ``id``, ``name``, ``description``
        and ``tags`` and nothing else, and the card sets its input and output
        modes at the agent level, so per-skill mode lists here were both unread
        and contradicting what was advertised. Adding a key that ``_build_card``
        does not read puts it nowhere.
        """
        return [
            {
                "id": s.name,
                "name": s.name,
                "description": s.description,
                "tags": [],
            }
            for s in self._skills.values()
        ]
