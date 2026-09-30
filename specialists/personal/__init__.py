"""Personal agent service.

Exposes the :class:`PersonalAgent` and the factory the application uses to
build its MCP tool server, and the :class:`ProfileStore` through which it
reaches the two columns of the user's record it owns. The row itself and the
conversation in it belong to :mod:`erebots.persistence`; the profile and the
consents belong here, and no other agent touches them.
"""

from specialists.personal.agent import PersonalAgent
from specialists.personal.store import ProfileStore
from specialists.personal.tools import build_personal_mcp, make_personal_tools

__all__ = ["PersonalAgent", "ProfileStore", "build_personal_mcp", "make_personal_tools"]
