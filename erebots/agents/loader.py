"""Agent configuration loader.

Reads ``agents.yaml`` and discovers agents via their A2A agent cards.
All agents are treated uniformly — the orchestrator only needs
``name``, ``description``, and ``a2a_url`` to route requests.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

from erebots.config import settings

logger = logging.getLogger(__name__)

TokenProvider = Callable[[], Awaitable[str]]


async def _fetch_agent_card(
    a2a_url: str,
    token_provider: TokenProvider | None = None,
) -> dict:
    """Fetch the agent card from an A2A server.

    Args:
        a2a_url: Base URL of the A2A server.
        token_provider: Optional async callable returning a bearer
            token; needed when the A2A endpoint requires M2M auth.

    Returns:
        Parsed agent card dictionary.

    Raises:
        httpx.HTTPError: On network or HTTP errors.
    """
    card_url = f"{a2a_url.rstrip('/')}/.well-known/agent-card.json"
    headers: dict[str, str] = {}
    if token_provider is not None:
        token = await token_provider()
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(card_url, headers=headers)
        response.raise_for_status()
        return response.json()


def configured_urls(config_path: str | Path | None = None) -> dict[str, str]:
    """Map agent name to the URL ``agents.yaml`` publishes it at.

    The name is the last path segment, which ``specialists/registry.py`` already
    requires to match the mount path. This is the same file the orchestrator
    dials, so a locally hosted agent has exactly one place declaring its
    address — and the card it serves can advertise that value instead of a
    second, separately maintained copy of it.

    Returns an empty mapping when the file is missing or unreadable; callers
    fall back to their own default.
    """
    try:
        targets = _configured_a2a_urls(config_path)
    except Exception:
        logger.exception("Failed to read %s", config_path or settings.agents_config_path)
        return {}

    urls: dict[str, str] = {}
    for url in targets:
        name = url.rsplit("/", 1)[-1]
        if url and name:
            urls[name] = url
    return urls


def _declared_endpoint(card: dict, fetched_from: str) -> str:
    """Return the operational endpoint the card declares, or the fetch URL.

    A2A cards advertise where to send messages under ``supportedInterfaces``,
    which is not necessarily where the card was fetched: an agent behind a proxy
    knows its public address, and may expose several bindings. Following the
    declaration is what makes discovery real rather than a lookup of something
    the caller already knew.

    The fetch URL remains the fallback, because a misconfigured peer that
    declares an unusable address should not become unreachable when we could
    still talk to it — we log the divergence instead of failing on it.
    """
    for interface in card.get("supportedInterfaces") or []:
        if not isinstance(interface, dict):
            continue
        url = str(interface.get("url") or "").strip().rstrip("/")
        if not url:
            continue
        if url != fetched_from:
            logger.info(
                "Agent card at %s declares a different endpoint (%s) — using the "
                "declared one",
                fetched_from,
                url,
            )
        return url
    logger.warning(
        "Agent card at %s declares no usable endpoint — falling back to the "
        "configured URL",
        fetched_from,
    )
    return fetched_from


@dataclass
class AgentSkill:
    """A single skill advertised on an agent card."""

    name: str
    description: str


@dataclass
class AgentEntry:
    """An agent discovered via its A2A agent card."""

    name: str
    description: str
    a2a_url: str
    skills: list[AgentSkill] = field(default_factory=list)


@dataclass
class LoadedAgents:
    """Result of loading the agents configuration.

    ``failed`` lists the configured URLs whose card could not be fetched or was
    unusable, so a caller can retry exactly those later.
    """

    entries: list[AgentEntry] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _configured_a2a_urls(config_path: str | Path | None = None) -> list[str]:
    """Read application config, defaulting to settings.agents_config_path.

    Relative paths resolve from the working directory, never the installed
    package. A caller can pass an absolute path to run from any directory.
    """
    path = Path(config_path if config_path is not None else settings.agents_config_path).expanduser()
    if not path.exists():
        logger.warning("Agent configuration not found at %s — no agents loaded", path)
        return []
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    urls: list[str] = []
    for agent_def in (config or {}).get("agents") or []:
        # Each entry is a plain A2A URL (a dict with ``a2a_url`` is also accepted).
        a2a_url = (
            agent_def if isinstance(agent_def, str) else agent_def.get("a2a_url", "")
        ).rstrip("/")
        if not a2a_url:
            logger.warning("Agent entry missing an A2A URL — skipping: %s", agent_def)
            continue
        urls.append(a2a_url)
    return urls


async def load_agents(
    token_provider: TokenProvider | None = None,
    urls: Iterable[str] | None = None,
    *,
    config_path: str | Path | None = None,
) -> LoadedAgents:
    """Load agents from ``agents.yaml`` by fetching their agent cards.

    Each entry must have an ``a2a_url``. Name and description are
    retrieved from ``<a2a_url>/.well-known/agent-card.json``.

    Args:
        token_provider: Optional async callable returning a bearer
            token, forwarded to each agent-card request.
        urls: Fetch these URLs instead of the file's — how a caller retries
            the ones a previous call reported in ``failed``. An empty iterable
            explicitly disables discovery; it does not fall back to YAML.
        config_path: Application YAML path. Defaults to the configured
            ``agents_config_path`` (``agents.yaml`` in the working directory).
            Ignored when ``urls`` is supplied.

    Returns:
        :class:`LoadedAgents` with the discovered agents and the URLs that
        could not be discovered.
    """
    targets = list(urls) if urls is not None else _configured_a2a_urls(config_path)
    result = LoadedAgents()

    for a2a_url in targets:
        try:
            card = await _fetch_agent_card(a2a_url, token_provider=token_provider)
        except Exception:
            logger.exception("Failed to fetch agent card from %s", a2a_url)
            result.failed.append(a2a_url)
            continue

        name = card.get("name", "")
        if not name:
            logger.error("Agent card at %s has no name — skipping", a2a_url)
            result.failed.append(a2a_url)
            continue

        skills: list[AgentSkill] = []
        for raw in card.get("skills") or []:
            if not isinstance(raw, dict):
                continue
            skill_name = str(raw.get("name") or raw.get("id") or "").strip()
            if not skill_name:
                continue
            skills.append(AgentSkill(
                name=skill_name,
                description=str(raw.get("description") or "").strip(),
            ))

        entry = AgentEntry(
            name=name,
            description=card.get("description", ""),
            a2a_url=_declared_endpoint(card, a2a_url),
            skills=skills,
        )
        result.entries.append(entry)
        logger.info(
            "Discovered agent '%s' at %s (%d skills)",
            # The resolved endpoint, not the one we fetched from: those differ
            # whenever the card declares its own address, and logging the fetch
            # URL here would point at somewhere no message is ever sent.
            name, entry.a2a_url, len(skills),
        )

    return result
