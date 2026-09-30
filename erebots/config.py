"""Application configuration module.

Loads environment variables from the ``.env`` file and exposes them
through the immutable :class:`Settings` dataclass. Each variable is
converted to the appropriate Python type (bool, int, list, str) using
dedicated helper functions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_env_file() -> None:
    """Load the application's ``.env`` into ``os.environ``.

    The file belongs to the *application*, not to this package, so it is looked
    for in the current working directory — the same rule ``agents_config_path``
    already follows. Resolving it relative to ``__file__`` (as this did) points
    at ``site-packages/.env`` once the framework is installed as a wheel, which
    never exists, so an installed consumer's configuration was silently ignored.
    ``EREBOTS_ENV_FILE`` overrides the location for anything that keeps its
    configuration elsewhere.

    Variables are only set if not already present in the environment
    (*setdefault* behavior), so a real environment variable always wins over the
    file. Blank lines and comments (``#``) are ignored; values may be enclosed in
    single or double quotes, which are stripped automatically.

    Setting ``EREBOTS_IGNORE_DOTENV`` skips the file entirely, so a process can
    run against the declared defaults regardless of what happens to sit on the
    developer's disk. The test suite sets it (see the root ``conftest.py``): CI
    runs on a clean checkout with no ``.env`` at all, and this makes a local run
    reproduce that same baseline instead of inheriting a local configuration.
    """
    if os.getenv("EREBOTS_IGNORE_DOTENV"):
        return
    env_path = Path(os.getenv("EREBOTS_ENV_FILE") or Path.cwd() / ".env")
    if not env_path.is_file():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        # Skip blank lines, comments, and lines without '='
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        # Strip whitespace and surrounding quotes from the value
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_bool(name: str, default: bool) -> bool:
    """Read an environment variable and convert it to a boolean.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set.

    Returns:
        ``True`` if the value is one of ``1``, ``true``, ``yes``, ``on``
        (case-insensitive), otherwise ``False``.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """Read an environment variable and convert it to an integer.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set or not convertible.

    Returns:
        The integer value of the variable, or *default* on absence
        or conversion error.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_list(name: str, default: list[str]) -> list[str]:
    """Read an environment variable and convert it to a list of strings.

    Supports two formats:
    - **JSON array**: ``["a", "b", "c"]``
    - **Comma-separated values**: ``a, b, c``

    Args:
        name: Name of the environment variable.
        default: Default list if the variable is not set or the format
            is invalid.

    Returns:
        The list of strings parsed from the variable.
    """
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    # Attempt JSON parsing if the value starts with '['
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    # Otherwise split by comma
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_str(name: str, default: str) -> str:
    """Read an environment variable as a string.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set.

    Returns:
        The variable value, stripped of whitespace and surrounding quotes.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().strip('"').strip("'")


# Public aliases: application-level config modules (e.g. the example app's
# ``specialists/config.py``) reuse these typed env readers for their own
# settings, keeping domain configuration out of the framework.
get_bool = _get_bool
get_int = _get_int
get_list = _get_list
get_str = _get_str


# Load environment variables before instantiating Settings
_load_env_file()


@dataclass(frozen=True)
class Settings:
    """Global application configuration (immutable).

    Field defaults are read from the environment when this class is defined
    (import time, right after ``.env`` is loaded), so ``Settings()`` returns
    the same values for the life of the process; tests derive variants with
    ``dataclasses.replace``. The dataclass is *frozen* to guarantee
    immutability after initialization.

    Attributes:
        keycloak_url: Base Keycloak URL.
        keycloak_public_url: Public Keycloak URL (for browser redirects).
        keycloak_internal_url: Internal Keycloak URL (server-to-server communication).
        keycloak_realm: Keycloak realm name.
        keycloak_client_id: OIDC client ID registered in Keycloak.
        keycloak_client_secret: OIDC client secret (optional).
        app_callback_url: OAuth2 callback URL.
        enable_auth: Enable or disable authentication.
        jwt_cookie_name: Cookie name for the JWT access token.
        id_token_cookie_name: Cookie name for the ID token.
        jwt_cookie_secure: *Secure* flag on JWT cookies.
        app_host: Host to start the server on.
        app_port: Port to start the server on.
        brand_name: Name shown in the UI and on the Keycloak login page.
        ui_path: Path the chat UI is mounted at; post-login redirects land here.
        frontend: Which chat UI the example app serves (``gradio`` or ``web``).
        orchestrator_api_key: API key for the orchestrator agent.
        orchestrator_api_base_url: Base URL of the orchestrator API.
        orchestrator_model: LLM model used by the orchestrator.
        orchestrator_thinking: Enable thinking mode for the orchestrator.
        agents_config_path: Path to the application's ``agents.yaml``.
        cors_origins: List of allowed origins for CORS requests.
        tool_loop_max_iterations: Maximum tool-calling rounds per turn.
        chat_history_max_messages: Most recent history messages forwarded to
            the LLM per turn; ``0`` disables the cap.
        routing_include_skills: Include per-skill lines in the routing prompt.
        orchestrator_max_specialists: How many specialists one turn may fan out
            to; ``1`` keeps single-agent dispatch.
        multi_agent_composition: How several answers become one reply
            (``synthesis`` or ``sections``).
        multimodal_enabled: Whether the configured models accept image parts;
            images are stripped when false.
        enable_parser_retry: Retry a failed turn with the conversation
            flattened into a single prompt.
        a2a_persistent_tasks: Persist A2A tasks in the database instead of
            in memory.
        a2a_conformance_agent: Mount the TCK conformance endpoint at
            ``/a2a/tck``.
        a2a_push_allow_insecure_webhooks: Accept push webhooks the default
            policy (https + globally routable) would reject.
        a2a_parked_task_ttl: Seconds a task parked in ``input-required`` stays
            open before it is canceled; ``0`` disables expiry.
        database_url: SQLAlchemy async URL for the application database.
        internal_client_id: Confidential Keycloak client for service-to-service
            auth on ``/a2a/*`` and ``/tools/*``.
        internal_client_secret: That client's secret.
    """

    # Keycloak OIDC configuration
    keycloak_url: str = _get_str("keycloak_url", "http://localhost:8080")
    keycloak_public_url: str = _get_str("keycloak_public_url", keycloak_url)
    keycloak_internal_url: str = _get_str("keycloak_internal_url", keycloak_url)
    keycloak_realm: str = _get_str("keycloak_realm", "erebots")
    keycloak_client_id: str = _get_str("keycloak_client_id", "erebots-webapp")
    keycloak_client_secret: str | None = os.getenv("keycloak_client_secret")
    app_callback_url: str = _get_str("app_callback_url", "http://localhost:7860/api/auth/callback")

    # Authentication and cookies
    enable_auth: bool = _get_bool("enable_auth", True)
    jwt_cookie_name: str = _get_str("jwt_cookie_name", "access_token")
    id_token_cookie_name: str = _get_str("id_token_cookie_name", "id_token")
    # Secure by default: cookies are only sent over HTTPS. Override to
    # ``false`` via the environment only for plain-HTTP local development.
    jwt_cookie_secure: bool = _get_bool("jwt_cookie_secure", True)

    # Server
    app_host: str = _get_str("app_host", "0.0.0.0")
    app_port: int = _get_int("app_port", 7860)

    # Brand name shown in the UI (page logo/title). Also mirror it onto the
    # Keycloak realm's display name to brand the login page.
    brand_name: str = _get_str("brand_name", "EREBOTS")

    # Path the chat UI is mounted at ("/" serves it at the root; post-login
    # redirects land here too).
    ui_path: str = _get_str("ui_path", "/")

    # Which chat UI the example app serves at ui_path: "gradio" (in-process
    # Blocks) or "web" (the static Vue SPA in examples/web, same-origin
    # cookie auth). Both consume the same orchestrator core.
    frontend: str = _get_str("frontend", "gradio")

    # Orchestrator agent
    orchestrator_api_key: str = _get_str("orchestrator_api_key", "EMPTY")
    orchestrator_api_base_url: str = _get_str(
        "orchestrator_api_base_url", "http://localhost:8080/v1"
    )
    orchestrator_model: str = _get_str("orchestrator_model", "local")
    orchestrator_thinking: bool = _get_bool("orchestrator_thinking", False)

    # Where the orchestrator's agent registry (``agents.yaml``) lives. A
    # relative path resolves against the process working directory, so an
    # installed (non-checkout) deployment can point this anywhere.
    agents_config_path: str = _get_str("agents_config_path", "agents.yaml")

    # Allowed CORS origins
    cors_origins: list[str] = field(
        default_factory=lambda: _get_list(
            "cors_origins",
            ["http://localhost", "http://localhost:7860"],
        )
    )

    # Maximum tool-calling iterations per turn (LLM → tool → observe loop).
    tool_loop_max_iterations: int = _get_int("tool_loop_max_iterations", 3)

    # Most recent chat-history messages forwarded to the LLM per turn (a
    # user+assistant pair is 2). Caps prompt growth in long conversations no
    # matter how much history a client sends. 0 disables the cap.
    chat_history_max_messages: int = _get_int("chat_history_max_messages", 20)

    # Hard limits on one request to the SSE chat endpoint. The byte cap is
    # enforced before the body is parsed (413); the others by the request
    # schema (422). Image size is measured on the data URL as sent.
    chat_request_max_bytes: int = _get_int("chat_request_max_bytes", 8 * 1024 * 1024)
    chat_message_max_chars: int = _get_int("chat_message_max_chars", 20_000)
    chat_images_max_per_message: int = _get_int("chat_images_max_per_message", 4)
    chat_image_max_bytes: int = _get_int("chat_image_max_bytes", 5 * 1024 * 1024)

    # Include each agent's per-skill lines in the routing prompt. Rich prompts
    # help large routing models; with a small router and many skills, the
    # lean form (name + description only) can route more reliably. The agent
    # card always advertises every skill regardless of this flag.
    routing_include_skills: bool = _get_bool("routing_include_skills", True)

    # Seconds before an agent (A2A card) or MCP server that failed discovery is
    # tried again. The ones that answered are usable meanwhile; one that was
    # down at the first request is picked up when it comes back, no restart.
    discovery_retry_seconds: int = _get_int("discovery_retry_seconds", 30)

    # How many specialists one turn may fan out to. 1 keeps the single-agent
    # dispatch (the router picks at most one target); higher values let the
    # router return several agents, which then run in parallel and are composed
    # into one reply. Doubles as the kill switch for multi-agent orchestration.
    orchestrator_max_specialists: int = _get_int("orchestrator_max_specialists", 3)

    # How several specialist answers become one reply. ``synthesis`` has the
    # orchestrator merge them with an extra LLM call (reads as one answer);
    # ``sections`` concatenates them under a per-agent heading (no extra call,
    # no risk of the merge distorting what a specialist said).
    multi_agent_composition: str = _get_str("multi_agent_composition", "synthesis")

    # Whether the configured models accept image parts. Set to false when any
    # backend in the deployment is text-only: image content is then stripped
    # before every request and replaced by a placeholder, so a text-only model
    # answers instead of failing on content it cannot parse. The framework does
    # not probe the model for this — it is a property of the deployment.
    multimodal_enabled: bool = _get_bool("multimodal_enabled", True)

    # Opt-in retry that flattens the conversation into a single prompt when the
    # backend raises a chat-template parser error (seen with vLLM/Qwen).
    # Off by default; harmless with llama.cpp/most backends.
    enable_parser_retry: bool = _get_bool("enable_parser_retry", False)

    # Persist A2A tasks in the app database instead of in memory. Needed only to
    # run multiple API instances behind a load balancer; off by default.
    a2a_persistent_tasks: bool = _get_bool("a2a_persistent_tasks", False)

    # Mount the A2A conformance endpoint at /a2a/tck, which replays the fixed
    # payloads the a2a-tck suite asks for by messageId. Enable it to run the
    # suite against this deployment; it answers from canned fixtures rather than
    # from an agent, so it has no reason to exist in a real one.
    a2a_conformance_agent: bool = _get_bool("a2a_conformance_agent", False)

    # Accept push-notification webhooks that plain HTTP or a non-public address
    # would otherwise disqualify. The agent POSTs the task's own content to that
    # URL, so the default policy (https + globally routable) is what keeps a
    # caller from aiming this server at its own network. Turn it on only to run
    # the conformance suite, whose receiver listens on http://localhost.
    a2a_push_allow_insecure_webhooks: bool = _get_bool("a2a_push_allow_insecure_webhooks", False)

    # Serve unauthenticated GET requests under /tools/* when they arrive from
    # loopback. Off by default: the MCP tool servers this framework mounts speak
    # Streamable HTTP over POST and send their own bearer, so they never need it.
    # It exists for a deployment that also mounts a *re-entrant* UI there — the
    # example app's Gradio tool servers call themselves back over loopback with
    # no Authorization header — and it is retired, not re-scoped, once those
    # migrate to FastMCP.
    #
    # Turning it on trusts the peer address: anything that can make a request
    # look like it came from loopback (a reverse proxy on the same host, a local
    # SSRF, a sidecar sharing the network namespace) can then read every GET
    # below /tools/. Only enable it where /tools/* is not routable from outside.
    allow_loopback_tool_reads: bool = _get_bool("allow_loopback_tool_reads", False)

    # Seconds a task parked in input-required (ask_user) stays open before it is
    # canceled. Resume answers a question within this window; past it, the user
    # has moved on and the parked task would otherwise stay open forever.
    # 0 disables expiry.
    a2a_parked_task_ttl: int = _get_int("a2a_parked_task_ttl", 900)

    # SQLAlchemy async URL for the application database (UserContext etc.).
    # Distinct from Keycloak's own DB (a separate Postgres instance).
    database_url: str = _get_str(
        "database_url",
        "postgresql+asyncpg://erebots:erebots@erebots-app-db:5432/erebots",
    )

    # Confidential Keycloak client used for service-to-service auth on
    # /a2a/* and /tools/* (client_credentials grant).
    internal_client_id: str = _get_str("internal_client_id", "erebots-internal")
    internal_client_secret: str = _get_str("internal_client_secret", "")


# Singleton configuration instance
settings = Settings()
