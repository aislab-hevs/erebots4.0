"""Webhook safety for A2A push notifications.

Push notifications invert who calls whom: the *server* POSTs to a URL the
*client* chose. That is a server-side request forgery primitive — the target
may be an address only this process can reach — and, because the payload
carries the task's own content, an exfiltration channel.

The specification asks agents to validate those URLs (§ Push Notification
Security: reject private ranges, loopback and link-local, prefer HTTPS); the
SDK's sender posts whatever it is given, so the check lives here. It runs when
a configuration is stored — where the caller can still be told why — and again
at delivery, where the connection is pinned to the re-validated address so a
DNS answering differently the second time gains nothing (see :func:`_pinned`).

``a2a_push_allow_insecure_webhooks`` turns the policy off. It exists for the
conformance suite, whose receiver listens on ``http://localhost``, and for
local development — never for a deployment reachable by another agent.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import SplitResult, urlsplit

import httpx
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    PushNotificationConfigStore,
    PushNotificationSender,
)
from a2a.utils.errors import InvalidParamsError
from a2a.utils.proto_utils import to_stream_response
from google.protobuf.json_format import MessageToDict

from erebots.config import settings

logger = logging.getLogger(__name__)

#: How long a webhook delivery may take. The spec suggests 10–30 seconds; a
#: receiver slower than this is not worth blocking the notification loop on.
WEBHOOK_TIMEOUT_SECONDS = 15.0


async def reject_unsafe_webhook(url: str) -> None:
    """Raise ``InvalidParamsError`` unless the agent may POST a task to ``url``.

    Every address the host resolves to must be globally routable: checking the
    string alone would let ``https://internal.example.com`` through when it
    answers 10.0.0.5. An IP literal is checked as it stands, without a lookup.

    Args:
        url: The webhook the client registered.

    Raises:
        InvalidParamsError: The URL is malformed, is not ``https``, has no
            host, does not resolve, or resolves to a non-public address.
    """
    if settings.a2a_push_allow_insecure_webhooks:
        return
    parsed = _split(url)
    if parsed.scheme != "https":
        raise InvalidParamsError(
            f"push notification url must use https, got {parsed.scheme or 'no scheme'}"
        )
    if not parsed.hostname:
        raise InvalidParamsError("push notification url has no host")
    for address in await _addresses_of(parsed.hostname):
        if not address.is_global:
            raise InvalidParamsError(
                f"push notification url resolves to the non-public address {address}"
            )


def _split(url: str) -> SplitResult:
    """Split ``url``, reporting a malformed one as ``InvalidParamsError``.

    ``urlsplit`` raises ``ValueError`` for a bad URL; the callers are both on
    request paths that must answer with a protocol error, not a 500.
    """
    try:
        return urlsplit(url)
    except ValueError as exc:  # malformed IPv6 literal, bad port, ...
        raise InvalidParamsError(f"push notification url is malformed: {exc}") from exc


async def _addresses_of(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Return every IP for ``host``, or the literal it already is.

    Raises:
        InvalidParamsError: The host does not resolve.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, None, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise InvalidParamsError(
            f"push notification host does not resolve: {host}"
        ) from exc
    # A scoped IPv6 result ("fe80::1%lo0") is not parseable — and is exactly the
    # kind of address the policy rejects, so drop the scope and judge the rest.
    return [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]


class ValidatedPushConfigStore(PushNotificationConfigStore):
    """A config store that refuses a webhook the agent must not call.

    Wrapping the store rather than the sender puts the refusal where the client
    is still listening: ``CreatePushNotificationConfig`` fails with a reason,
    instead of the notification silently never arriving.
    """

    def __init__(self, store: PushNotificationConfigStore) -> None:
        """Wrap ``store``, validating every webhook on its way in.

        Args:
            store: The store that actually holds the configurations.
        """
        self._store = store

    async def set_info(self, task_id, notification_config, context) -> None:
        """Store a configuration, once its webhook has passed validation.

        This is the only method that guards: the others read back what was
        already accepted here.

        Raises:
            InvalidParamsError: The agent must not call this webhook. Nothing
                is stored.
        """
        await reject_unsafe_webhook(notification_config.url)
        await self._store.set_info(task_id, notification_config, context)

    async def get_info(self, task_id, context):
        """Return the configurations registered for ``task_id`` (delegated)."""
        return await self._store.get_info(task_id, context)

    async def get_info_for_dispatch(self, task_id):
        """Return the configurations to notify for ``task_id`` (delegated)."""
        return await self._store.get_info_for_dispatch(task_id)

    async def delete_info(self, task_id, context, config_id=None) -> None:
        """Remove a configuration for ``task_id`` (delegated)."""
        await self._store.delete_info(task_id, context, config_id)


class AuthenticatedPushSender(PushNotificationSender):
    """Deliver notifications carrying the credentials the receiver asked for.

    The spec requires the agent to authenticate its webhook call as specified in
    ``PushNotificationConfig.authentication``. The SDK's sender forwards only the
    opaque ``token`` header and drops that field, so the receiver cannot tell a
    genuine notification from anything else that found its URL — hence this
    sender rather than the SDK's.

    Code with an expiry date: as of a2a-sdk 1.1.0 the word ``authentication``
    does not appear in ``base_push_notification_sender`` at all. When a release
    starts reading that field, delete this class and pass the SDK's
    ``BasePushNotificationSender`` to :func:`push_support` instead.
    """

    def __init__(
        self, client: httpx.AsyncClient, config_store: PushNotificationConfigStore
    ) -> None:
        """Bind the sender to the client it delivers with and the store it reads.

        Args:
            client: Shared webhook client (see :func:`webhook_client`).
            config_store: Where the per-task configurations are looked up —
                normally the :class:`ValidatedPushConfigStore` from
                :func:`push_support`.
        """
        self._client = client
        self._store = config_store

    async def send_notification(self, task_id: str, event) -> None:
        """POST ``event`` to every webhook registered for ``task_id``.

        Deliveries run concurrently and none of them can raise: a receiver
        that is down must not affect the task it is reporting on.

        Args:
            task_id: The task whose subscribers are notified.
            event: The task update to serialize into the notification body.
        """
        configs = await self._store.get_info_for_dispatch(task_id)
        if not configs:
            return
        payload = MessageToDict(to_stream_response(event))
        await asyncio.gather(
            *(self._deliver(config, payload, task_id) for config in configs)
        )

    async def _deliver(self, config, payload: dict, task_id: str) -> None:
        """Attempt one delivery without letting failure escape to the task.

        Every failure — re-validation, connection, non-2xx status — is logged
        and swallowed. There is no retry: the task's own state remains the
        authoritative record a client can poll.

        Args:
            config: The subscriber's push configuration.
            payload: The already-serialized notification body.
            task_id: Included in the log line when the delivery fails.
        """
        try:
            url, headers, extensions = await _pinned(config.url)
            headers.update(_delivery_headers(config))
            response = await self._client.post(
                url, json=payload, headers=headers, extensions=extensions
            )
            response.raise_for_status()
        except Exception:
            logger.exception(
                "push notification for task %s to %s failed", task_id, config.url
            )


async def _pinned(url: str) -> tuple[str, dict[str, str], dict]:
    """Re-validate ``url`` at delivery time and pin the connection to the result.

    The registration-time check resolved the host once; a DNS that answers
    differently the second time would route the delivery somewhere the policy
    never judged (rebinding). Resolving again here and connecting to that very
    address closes the window: the socket goes to the address that was checked,
    while the ``Host`` header and TLS SNI keep the original name so certificate
    verification still applies to it.

    Args:
        url: The registered webhook, about to be called.

    Returns:
        ``(request_url, headers, request_extensions)`` ready to pass to the
        client. When the policy is off, or the host is already an IP literal
        with nothing left to rebind, the URL comes back unchanged with nothing
        to add.

    Raises:
        InvalidParamsError: The host now answers a non-public address. Callers
            treat that as a failed delivery, never as a failed task.
    """
    if settings.a2a_push_allow_insecure_webhooks:
        return url, {}, {}
    parsed = _split(url)
    addresses = await _addresses_of(parsed.hostname)
    for address in addresses:
        if not address.is_global:
            raise InvalidParamsError(
                f"push notification url resolves to the non-public address {address}"
            )
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        # Already a (validated) IP literal: nothing left to rebind.
        return url, {}, {}
    host_header = parsed.hostname
    if parsed.port:
        host_header = f"{host_header}:{parsed.port}"
    return (
        str(httpx.URL(url).copy_with(host=str(addresses[0]))),
        {"Host": host_header},
        {"sni_hostname": parsed.hostname},
    )


def _delivery_headers(config) -> dict[str, str]:
    """Build the headers the receiver needs to trust the call.

    Both mechanisms travel when both are configured: the notification token
    identifies the subscription, ``Authorization`` authenticates the caller.

    Args:
        config: The subscriber's push configuration.

    Returns:
        The headers to add to the delivery — empty when the receiver asked
        for neither mechanism.
    """
    headers = {}
    if config.token:
        headers["X-A2A-Notification-Token"] = config.token
    auth = config.authentication
    if auth.scheme and auth.credentials:
        headers["Authorization"] = f"{auth.scheme} {auth.credentials}"
    return headers


#: One client for every webhook this process delivers. Kept module-level because
#: a mounted sub-app never receives lifespan events, so it has nowhere of its own
#: to close it; :func:`aclose_webhook_client` is called from the app's lifespan.
_client: httpx.AsyncClient | None = None


def webhook_client() -> httpx.AsyncClient:
    """Return the HTTP client used to deliver notifications.

    Redirects are not followed: a URL that passed validation could redirect to
    one that never would, and the check has already happened by then.

    Returns:
        The process-wide client, created on first use.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=WEBHOOK_TIMEOUT_SECONDS, follow_redirects=False
        )
    return _client


async def aclose_webhook_client() -> None:
    """Close the shared client (app shutdown); the next call builds a new one."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def push_support(
    config_store: PushNotificationConfigStore | None = None,
) -> tuple[PushNotificationConfigStore, AuthenticatedPushSender]:
    """The store/sender pair a request handler needs to serve push notifications.

    Args:
        config_store: Where configurations live (defaults to in-memory; pass a
            ``DatabasePushNotificationConfigStore`` to survive a restart). It is
            wrapped so no unvalidated webhook can reach it.

    Returns:
        The validated store and a sender bound to it.
    """
    store = ValidatedPushConfigStore(
        config_store or InMemoryPushNotificationConfigStore()
    )
    return store, AuthenticatedPushSender(webhook_client(), store)
