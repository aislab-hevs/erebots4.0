"""MCP (Model Context Protocol) package for remote tool discovery and invocation."""

from erebots.agents.mcp.client import MCPClient, MCPToolError

__all__ = ["MCPClient", "MCPToolError"]
