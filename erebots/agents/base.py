"""Abstract base agent with generative layer, skill registry, and optional MCP/A2A support."""

from __future__ import annotations

from abc import ABC, abstractmethod

from erebots.agents.generative import GenerativeLayer
from erebots.agents.models import AgentResponse
from erebots.agents.skill import SkillRegistry


class BaseAgent(ABC):
    """Abstract base agent with generative layer, skill registry, and optional MCP/A2A clients.

    Provides the common infrastructure for all agents: generative layer initialization,
    skill management, and methods for streaming and stopping. Subclasses must implement
    ``setup_skills`` and ``process``.
    """

    def __init__(
        self,
        agent_name: str,
        api_key: str,
        api_base_url: str,
        model: str,
        system_prompt: str,
        description: str = "",
        thinking: bool = False,
    ):
        """Initialize the base agent with the generative layer and skill registry.

        Args:
            agent_name: Unique identifier name of the agent.
            api_key: API key for the LLM service.
            api_base_url: Base URL of the OpenAI-compatible API endpoint.
            model: LLM model identifier to use.
            system_prompt: System prompt to prepend to conversations.
            description: Short human-readable description used by the
                orchestrator's router LLM to discriminate specialists.
            thinking: If True, enables the model's explicit reasoning mode.
        """
        self.name = agent_name
        self.description = description
        self.generative = GenerativeLayer(
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            system_prompt=system_prompt,
            thinking=thinking,
        )
        self.skills = SkillRegistry()
        self.mcp_client = None
        self.a2a_client = None

    @abstractmethod
    def setup_skills(self):
        """Register agent-specific skills.

        Must be implemented by subclasses to populate the ``self.skills``
        registry with the skills needed by the agent.
        """
        ...

    @abstractmethod
    async def process(self, message: str | list[dict], context: dict | None = None) -> AgentResponse:
        """Process a message and return a response with execution trace.

        Args:
            message: Text message or list of dictionaries (multi-modal format).
            context: Optional additional context for processing.

        Returns:
            ``AgentResponse`` object containing the response content and trace.
        """
        ...

    def chat_stream(self, messages: list[dict]):
        """Proxy streaming chat to the generative layer.

        Args:
            messages: List of conversation messages as dictionaries.

        Returns:
            Generator yielding text fragments of the streaming response.
        """
        return self.generative.chat_stream(messages)

    def stop(self):
        """Stop the current streaming by delegating to the generative layer."""
        self.generative.stop()
