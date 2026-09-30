"""A2A (Agent-to-Agent) package for communication between specialized agents."""

from erebots.agents.a2a.client import A2AClient
from erebots.agents.a2a.server import create_a2a_app

__all__ = ["A2AClient", "create_a2a_app"]
