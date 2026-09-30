"""Orchestrator agent package.

Exposes the ``OrchestratorAgent`` class, the main entry point
for dispatching requests to specialized agents.
"""

from erebots.agents.orchestrator.agent import OrchestratorAgent

__all__ = ["OrchestratorAgent"]
