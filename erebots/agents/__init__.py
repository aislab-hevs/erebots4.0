"""Agents package: base classes and components for building agents.

``BaseSpecialist`` is the extension point most applications reach for — it owns
MCP discovery and the tool-calling loop, so a domain agent contributes a prompt,
a description and its skills. See ``docs/api-stability.md``.
"""

from erebots.agents.base import BaseAgent
from erebots.agents.generative import GenerativeLayer
from erebots.agents.models import AgentResponse, InputRequired, TraceStep
from erebots.agents.skill import Skill, SkillRegistry
from erebots.agents.specialist import BaseSpecialist

__all__ = [
    "AgentResponse",
    "BaseAgent",
    "BaseSpecialist",
    "GenerativeLayer",
    "InputRequired",
    "Skill",
    "SkillRegistry",
    "TraceStep",
]
