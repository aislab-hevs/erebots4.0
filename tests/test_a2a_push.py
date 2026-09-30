"""Push notifications: webhook validation and delivery.

The agent POSTs the task's own content to a URL the caller chose, so these
tests are as much about what the server *refuses* to call as about what it
delivers. Requests are driven over the REST binding, the way the a2a-tck suite
exercises them.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount

from erebots.agents.a2a import push
from erebots.agents.a2a.conformance import create_conformance_app

MOUNT = "/a2a/tck"
BASE = "http://test"
VERSION = {"A2A-Version": "1.0"}

#: A public IP literal: exercises the policy without depending on DNS.
PUBLIC_WEBHOOK = "https://93.184.216.34/hook"


def _mounted_app() -> Starlette:
    return Starlette(routes=[Mount(MOUNT, app=create_conformance_app(url=BASE + MOUNT))])


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE,
        follow_redirects=True,
        headers=VERSION,
    )


async def _a_task(http: httpx.AsyncClient) -> str:
    """Send one message and return the id of the task it created."""
    response = await http.post(
        f"{MOUNT}/message:send",
        json={
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "hi"}],
                "messageId": "tck-complete-task-1",
            }
        },
    )
    return response.json()["task"]["id"]


def _allow_insecure(monkeypatch, allowed: bool) -> None:
    """Settings is frozen: swap the whole instance rather than a field."""
    monkeypatch.setattr(
        push,
        "settings",
        replace(push.settings, a2a_push_allow_insecure_webhooks=allowed),
    )


# --------------------------------------------------------------------------
# Capability + CRUD
# --------------------------------------------------------------------------


async def test_the_card_declares_push_notifications():
    """The capability gates every push operation: undeclared, they must error."""
    async with _client(_mounted_app()) as http:
        card = (await http.get(f"{MOUNT}/.well-known/agent-card.json")).json()

    assert card["capabilities"]["pushNotifications"] is True


async def test_a_public_https_webhook_is_accepted_and_readable():
    async with _client(_mounted_app()) as http:
        task_id = await _a_task(http)
        created = await http.post(
            f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs",
            json={"url": PUBLIC_WEBHOOK, "token": "s3cret"},
        )
        listed = await http.get(f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs")

    assert created.status_code == 200
    assert created.json()["url"] == PUBLIC_WEBHOOK
    assert [c["url"] for c in listed.json()["configs"]] == [PUBLIC_WEBHOOK]


async def test_deleting_a_config_twice_is_not_an_error():
    """PUSH-DEL-002: delete is idempotent — the second call still succeeds."""
    async with _client(_mounted_app()) as http:
        task_id = await _a_task(http)
        created = await http.post(
            f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs",
            json={"url": PUBLIC_WEBHOOK},
        )
        path = f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs/{created.json()['id']}"
        first = await http.delete(path)
        second = await http.delete(path)

    assert (first.status_code, second.status_code) == (200, 200)


# --------------------------------------------------------------------------
# What the agent refuses to call
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("webhook", "because"),
    [
        ("http://93.184.216.34/hook", "must use https"),
        ("https://127.0.0.1/hook", "non-public"),
        ("https://10.0.0.5/hook", "non-public"),
        ("https://[::1]/hook", "non-public"),
        ("https:///hook", "no host"),
    ],
)
async def test_an_unsafe_webhook_is_refused_with_a_reason(webhook, because):
    """Refusing at configuration time is what lets the caller learn why.

    Accepting it and failing at delivery would leave the client waiting for a
    notification that is never coming — and would have already made the call.
    """
    async with _client(_mounted_app()) as http:
        task_id = await _a_task(http)
        response = await http.post(
            f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs", json={"url": webhook}
        )

    assert response.status_code == 400
    assert because in response.json()["error"]["message"]


async def test_a_hostname_answering_a_private_address_is_refused(monkeypatch):
    """The check resolves the host: a name is not evidence of where it points.

    Validating the string alone lets an internal host through under a public
    name — the shape SSRF filters are usually bypassed with.
    """
    import socket

    def _answers_privately(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 443))]

    # The loop delegates to socket.getaddrinfo, which is the resolver to fake.
    monkeypatch.setattr(socket, "getaddrinfo", _answers_privately)

    with pytest.raises(Exception) as raised:
        await push.reject_unsafe_webhook("https://looks-public.example.com/hook")

    assert "10.1.2.3" in str(raised.value)


async def test_the_policy_can_be_turned_off_for_local_testing(monkeypatch):
    """The conformance suite's receiver listens on http://localhost.

    Without an escape hatch the three delivery requirements are untestable; the
    setting is documented as belonging to a test run, not to a deployment.
    """
    _allow_insecure(monkeypatch, True)

    await push.reject_unsafe_webhook("http://localhost:9999/webhook")


# --------------------------------------------------------------------------
# Delivery
# --------------------------------------------------------------------------


async def test_a_notification_reaches_the_webhook_with_its_token(monkeypatch):
    """PUSH-DELIVER-001/003: the agent calls the webhook, authenticated.

    The shared client is swapped for one that records instead of connecting, so
    the assertion is on what would go over the wire.
    """
    delivered: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        delivered.append(request)
        return httpx.Response(200)

    _allow_insecure(monkeypatch, True)
    monkeypatch.setattr(
        push, "_client", httpx.AsyncClient(transport=httpx.MockTransport(_record))
    )

    async with _client(_mounted_app()) as http:
        # Park a task first: a completed one has no further update to announce.
        parked = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "hi"}],
                    "messageId": "tck-input-required-1",
                }
            },
        )
        task_id = parked.json()["task"]["id"]
        await http.post(
            f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs",
            json={
                "url": "http://localhost:9999/webhook",
                "token": "s3cret",
                "authentication": {"scheme": "Bearer", "credentials": "t0ken"},
            },
        )
        # Answering it drives the task to completion — the update to notify.
        await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "again"}],
                    "messageId": "tck-complete-task-2",
                    "taskId": task_id,
                }
            },
        )

    assert delivered, "no webhook delivery was attempted"
    request = delivered[0]
    assert str(request.url) == "http://localhost:9999/webhook"
    assert request.headers["X-A2A-Notification-Token"] == "s3cret"
    # PUSH-DELIVER-001: the receiver must be able to tell this call is genuine.
    # The SDK's own sender drops `authentication` and sends only the token.
    assert request.headers["Authorization"] == "Bearer t0ken"
    # PUSH-DELIVER-003: a StreamResponse carries exactly one of these.
    payload = json.loads(request.content)
    assert len(payload) == 1
    assert next(iter(payload)) in {"task", "message", "statusUpdate", "artifactUpdate"}


# --------------------------------------------------------------------------
# Delivery pinning (DNS rebinding)
# --------------------------------------------------------------------------


def _resolves_to(monkeypatch, address: str) -> None:
    """Make every hostname resolve to ``address`` for both validation points."""
    import ipaddress

    async def _fixed(host):
        return [ipaddress.ip_address(address)]

    monkeypatch.setattr(push, "_addresses_of", _fixed)


async def _drive_a_delivery(webhook: str) -> None:
    """Park a task, register ``webhook``, answer — which triggers the delivery."""
    async with _client(_mounted_app()) as http:
        parked = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "hi"}],
                    "messageId": "tck-input-required-1",
                }
            },
        )
        task_id = parked.json()["task"]["id"]
        await http.post(
            f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs", json={"url": webhook}
        )
        await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "again"}],
                    "messageId": "tck-complete-task-2",
                    "taskId": task_id,
                }
            },
        )


async def test_delivery_connects_to_the_address_that_was_validated(monkeypatch):
    """The socket must go where the check looked, not where DNS points *now*.

    Registration resolves the host once; delivering by name would resolve it
    again, and a DNS answering differently the second time (rebinding) routes
    the task's content somewhere the policy never judged. Pinning means the
    request URL carries the validated IP while Host and SNI keep the name.
    """
    delivered: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        delivered.append(request)
        return httpx.Response(200)

    _resolves_to(monkeypatch, "93.184.216.34")
    monkeypatch.setattr(
        push, "_client", httpx.AsyncClient(transport=httpx.MockTransport(_record))
    )

    await _drive_a_delivery("https://hook.example.com/hook")

    assert delivered, "no webhook delivery was attempted"
    request = delivered[0]
    assert request.url.host == "93.184.216.34"
    assert request.headers["Host"] == "hook.example.com"
    # TLS still negotiates (and verifies the certificate) for the *name*.
    assert request.extensions.get("sni_hostname") == "hook.example.com"


async def test_delivery_is_refused_when_the_host_has_moved_to_a_private_address(
    monkeypatch,
):
    """The rebinding scenario itself: validated public, delivered private — no.

    The config passed validation when the host answered a public address; by
    delivery time it answers 10.0.0.5. The delivery must be dropped (and the
    task must not fail), because this is exactly the swap the attacker makes.
    """
    delivered: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        delivered.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(
        push, "_client", httpx.AsyncClient(transport=httpx.MockTransport(_record))
    )
    _resolves_to(monkeypatch, "93.184.216.34")  # what the registration check saw

    async with _client(_mounted_app()) as http:
        parked = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "hi"}],
                    "messageId": "tck-input-required-1",
                }
            },
        )
        task_id = parked.json()["task"]["id"]
        registered = await http.post(
            f"{MOUNT}/tasks/{task_id}/pushNotificationConfigs",
            json={"url": "https://hook.example.com/hook"},
        )
        assert registered.status_code == 200

        _resolves_to(monkeypatch, "10.0.0.5")  # DNS answer changes underneath
        answered = await http.post(
            f"{MOUNT}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "again"}],
                    "messageId": "tck-complete-task-2",
                    "taskId": task_id,
                }
            },
        )

    assert not delivered, "the rebound webhook must not be called"
    # The failed delivery must never fail the task itself.
    assert answered.json()["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


async def test_an_ip_literal_webhook_is_delivered_as_it_stands(monkeypatch):
    """No name, no rebinding: the literal was validated and is dialed directly."""
    delivered: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        delivered.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(
        push, "_client", httpx.AsyncClient(transport=httpx.MockTransport(_record))
    )

    await _drive_a_delivery(PUBLIC_WEBHOOK)

    assert delivered and str(delivered[0].url) == PUBLIC_WEBHOOK
    assert delivered[0].extensions.get("sni_hostname") is None


def test_the_webhook_client_never_follows_a_redirect():
    """A validated URL could redirect to one that would not have passed, and by
    then the check is behind us."""
    client = push.webhook_client()

    assert client.follow_redirects is False
    assert client.timeout.connect == push.WEBHOOK_TIMEOUT_SECONDS
