"""Conversion agent service.

Exposes the :class:`ConversionAgent` and the factory building its MCP tool
server. The server is stateless, so the factory takes no arguments.
"""

from specialists.conversion.agent import ConversionAgent
from specialists.conversion.tools import build_conversion_mcp, convert
