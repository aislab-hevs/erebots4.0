"""MCP client for discovering and invoking tools on remote servers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

TokenProvider = Callable[[], Awaitable[str]]


class MCPToolError(RuntimeError):
    """A tool ran and reported failure, as opposed to being unreachable.

    The distinction is the reason this type exists: a transport error may be
    worth retrying, a tool that refused its arguments never is. Both used to
    arrive as ordinary text, which left a caller pattern-matching English prose
    to tell an answer from a failure.

    The message is the server's own, unmodified, so it can be handed to a model
    as the observation for that tool call.

    Attributes:
        tool: Name of the tool that reported the failure.
    """

    def __init__(self, tool: str, message: str):
        super().__init__(message)
        self.tool = tool


class MCPClient:
    """MCP client with dictionary-based tool discovery.

    Manages a local registry of MCP tools and allows discovering new
    tools from remote servers and invoking them via the MCP protocol.
    """

    def __init__(self, token_provider: TokenProvider | None = None):
        """Initialize the client.

        Each client owns its registry — clients never share tool state, so one
        agent's discovered tools cannot leak into another's.

        Args:
            token_provider: Optional async callable returning a bearer token,
                sent as the ``Authorization`` header on every MCP request.
        """
        self._registry: dict[str, dict] = {}
        self._openai_tools: list[dict] = []
        self.token_provider = token_provider

    async def _auth_headers(self) -> dict[str, str]:
        """Build the auth headers, fetching a fresh token if configured."""
        if self.token_provider is None:
            return {}
        token = await self.token_provider()
        return {"Authorization": f"Bearer {token}"}

    def register_tool(self, name: str, url: str, description: str = "", input_schema: dict | None = None):
        """Manually register a tool in the local registry.

        Args:
            name: Identifier name of the tool.
            url: URL of the MCP server hosting the tool.
            description: Text description of the tool.
            input_schema: JSON schema of the accepted input parameters.
        """
        self._registry[name] = {"url": url, "description": description, "input_schema": input_schema or {}}
        # Keep the OpenAI view in step with the registry. Without this a caller
        # that registers a tool and then asks for ``get_openai_tools`` gets an
        # empty list and no error -- both are public, so needing a private
        # rebuild call between them was a trap rather than a contract.
        self._build_openai_tools()

    def list_tools(self) -> list[dict]:
        """Return the list of all registered tools with their metadata.

        Returns:
            List of dictionaries with name, URL, description, and schema for each tool.
        """
        return [
            {"name": name, **info}
            for name, info in self._registry.items()
        ]

    def get_openai_tools(self) -> list[dict]:
        """Return tools in OpenAI function-calling format.

        Returns:
            List of dictionaries compatible with the OpenAI function-calling API.
        """
        return self._openai_tools

    def _build_openai_tools(self):
        """Rebuild the OpenAI tools list from the registry."""
        self._openai_tools = []
        for name, info in self._registry.items():
            schema = info.get("input_schema", {})
            self._openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.get("description", ""),
                    "parameters": schema if schema else {"type": "object", "properties": {}},
                },
            })

    async def discover_tools(self, server_url: str) -> list[dict]:
        """Discover available tools on a remote MCP server.

        Connects to the server via streamable HTTP, initializes the session,
        and retrieves the list of exposed tools.

        Args:
            server_url: URL of the MCP server to query.

        Returns:
            List of dictionaries with name, description, and input schema
            for each discovered tool.

        Raises:
            Exception: Transport, protocol and authentication failures
                propagate; callers discovering from several servers catch per
                server so one unreachable server does not cost them the rest.
        """
        headers = await self._auth_headers()
        async with streamablehttp_client(url=server_url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema,
                    }
                    for t in result.tools
                ]

    async def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Invoke a tool by name via its registered MCP server.

        Connects to the MCP server registered for the tool, initializes
        the session, and executes the call with the provided arguments.

        Args:
            name: Name of the tool to invoke.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            Text result from tool execution, obtained by concatenating
            the text parts of the response.

        Raises:
            ValueError: If the tool is not present in the registry.
            MCPToolError: If the tool ran and reported failure.
            Exception: Transport and protocol failures propagate.
        """
        tool_info = self._registry.get(name)
        if not tool_info:
            raise ValueError(f"Tool '{name}' not found in registry")

        server_url = tool_info["url"]
        headers = await self._auth_headers()
        async with streamablehttp_client(url=server_url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    name=name,
                    arguments=arguments or {},
                )
                parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        parts.append(content.text)
                text = "\n".join(parts)
                failed = result.isError

        # Raised out here, after both context managers have closed, and that is
        # load-bearing: ``ClientSession`` runs inside an anyio task group, so an
        # exception raised within the ``async with`` escapes as an ExceptionGroup
        # wrapping it. Callers catching MCPToolError would miss it entirely and
        # fall through to their generic handler.
        #
        # The check itself is the point: a tool that raises comes back as a normal
        # result carrying the error text, so without it a failure is
        # indistinguishable from an answer.
        if failed:
            raise MCPToolError(name, text or f"Tool {name!r} failed without a message.")
        return text
