"""BaseSpecialist MCP discovery: a server that is down is retried, not forgotten.

Discovery used to latch as soon as one server answered, so a second server
that was briefly unreachable stayed without tools until a restart.
"""

from __future__ import annotations

from erebots.agents.specialist import BaseSpecialist


class _StubSpecialist(BaseSpecialist):
    def setup_skills(self):
        pass


class _MCP:
    """Records discovery calls; ``down`` is the set of URLs that fail."""

    def __init__(self, down: set[str]):
        self.down = down
        self.discovered: list[str] = []
        self.registered: list[tuple[str, str]] = []
        self.builds = 0
        self.token_provider = None

    async def discover_tools(self, url):
        self.discovered.append(url)
        if url in self.down:
            raise ConnectionError("down")
        return [{"name": f"tool@{url}", "description": "", "input_schema": {}}]

    def register_tool(self, name, url, description, input_schema):
        self.registered.append((name, url))

    def _build_openai_tools(self):
        self.builds += 1


def _specialist(down: set[str]) -> tuple[BaseSpecialist, _MCP]:
    sp = _StubSpecialist(
        agent_name="stub",
        api_key="EMPTY",
        api_base_url="http://x/v1",
        model="local",
        system_prompt="s",
        description="stub",
        mcp_server_urls=["http://a/mcp", "http://b/mcp"],
    )
    mcp = _MCP(down)
    sp.mcp_client = mcp
    return sp, mcp


async def test_a_server_that_fails_discovery_is_retried_after_the_throttle():
    sp, mcp = _specialist(down={"http://b/mcp"})

    await sp._ensure_tools_discovered()
    assert mcp.registered == [("tool@http://a/mcp", "http://a/mcp")]
    assert sp._mcp_pending == ["http://b/mcp"]
    assert mcp.builds == 1

    # Throttled: the next turn does not hammer the server that just failed.
    await sp._ensure_tools_discovered()
    assert mcp.discovered == ["http://a/mcp", "http://b/mcp"]

    # Window over and the server is back: only the missing one is retried.
    mcp.down.clear()
    sp._mcp_retry_after = 0.0
    await sp._ensure_tools_discovered()
    assert mcp.discovered == ["http://a/mcp", "http://b/mcp", "http://b/mcp"]
    assert ("tool@http://b/mcp", "http://b/mcp") in mcp.registered
    assert sp._mcp_pending == []
    assert mcp.builds == 2

    # Nothing pending: discovery is over.
    sp._mcp_retry_after = 0.0
    await sp._ensure_tools_discovered()
    assert len(mcp.discovered) == 3


async def test_all_servers_down_keeps_every_url_pending_and_builds_nothing():
    sp, mcp = _specialist(down={"http://a/mcp", "http://b/mcp"})
    await sp._ensure_tools_discovered()
    assert sp._mcp_pending == ["http://a/mcp", "http://b/mcp"]
    assert mcp.builds == 0


async def test_no_servers_means_no_discovery():
    sp = _StubSpecialist(
        agent_name="stub",
        api_key="EMPTY",
        api_base_url="http://x/v1",
        model="local",
        system_prompt="s",
        description="stub",
    )
    assert not sp._mcp_discovery_due()
