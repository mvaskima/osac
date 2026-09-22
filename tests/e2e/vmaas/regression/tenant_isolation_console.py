from __future__ import annotations

import contextlib
import logging
import re
import ssl
import time
from collections.abc import Callable

import websocket

from tests.e2e.core.grpc_client import GRPCClient

logger = logging.getLogger(__name__)

CONSOLE_WS_PATH = "/api/fulfillment/v1/console_sessions/connect"
CONSOLE_USER = "fedora"
CONSOLE_PASSWORD = "osace2e"

_LOGIN_TIMEOUT_S = 300.0
_AUTH_TIMEOUT_S = 60.0
_PING_TIMEOUT_S = 40.0
_ENTER_INTERVAL_S = 15.0


def user_data(*, username: str = CONSOLE_USER, password: str = CONSOLE_PASSWORD) -> str:
    """Cloud-init userdata that sets a serial-console password for guest ping."""
    return (
        "#cloud-config\n"
        "ssh_pwauth: true\n"
        "chpasswd:\n"
        "  expire: false\n"
        "  users:\n"
        f"    - name: {username}\n"
        f"      password: {password}\n"
        "      type: text\n"
    )


def ping(
    *,
    grpc: GRPCClient,
    fulfillment_address: str,
    vm_id: str,
    dest_ip: str,
    username: str = CONSOLE_USER,
    password: str = CONSOLE_PASSWORD,
) -> bool:
    """Log in on the VM serial console and run ``ping -c 3 -W 2`` to dest_ip.

    Opens a new console ticket for each call and closes the session afterwards
    (tickets are single-use; a second concurrent session is rejected).
    """
    session = grpc.create_console_session(
        resource_type="CONSOLE_RESOURCE_TYPE_COMPUTE_INSTANCE", resource_id=vm_id, console_type="CONSOLE_TYPE_SERIAL"
    )
    ticket = session["ticket"]
    url = _ws_url(fulfillment_address)
    ws = _ws_connect(url, ticket)
    try:
        _wait_for_login_prompt(send=ws.send_binary, recv=lambda timeout, _ws=ws: _ws_recv(_ws, timeout))
        _authenticate(
            send=ws.send_binary,
            recv=lambda timeout, _ws=ws: _ws_recv(_ws, timeout),
            username=username,
            password=password,
        )
        return _run_ping(send=ws.send_binary, recv=lambda timeout, _ws=ws: _ws_recv(_ws, timeout), dest_ip=dest_ip)
    finally:
        with contextlib.suppress(Exception):
            ws.close()
            ws.shutdown()


def _ws_url(fulfillment_address: str) -> str:
    host: str = fulfillment_address.rsplit(":", 1)[0]
    return f"wss://{host}{CONSOLE_WS_PATH}"


def _ws_connect(url: str, ticket: str, timeout: int = 30) -> websocket.WebSocket:
    return websocket.create_connection(
        url,
        header={"Authorization": f"Bearer {ticket}"},
        sslopt={"cert_reqs": ssl.CERT_NONE},
        subprotocols=["binary"],
        timeout=timeout,
    )


def _ws_recv(ws: websocket.WebSocket, timeout: float) -> str | None:
    ws.settimeout(timeout)
    try:
        data = ws.recv()
        if isinstance(data, bytes):
            return data.decode(errors="replace")
        return data if data else None
    except websocket.WebSocketTimeoutException:
        return None


def _wait_for_login_prompt(
    *,
    send: Callable[[bytes], None],
    recv: Callable[[float], str | None],
    timeout: float = _LOGIN_TIMEOUT_S,
    enter_interval: float = _ENTER_INTERVAL_S,
) -> str:
    accumulated = ""
    deadline = time.monotonic() + timeout
    next_enter = time.monotonic()
    while time.monotonic() < deadline:
        if time.monotonic() >= next_enter:
            send(b"\n")
            logger.info("Sent enter to console")
            next_enter = time.monotonic() + enter_interval
        remaining = deadline - time.monotonic()
        chunk = recv(min(remaining, 5.0))
        if chunk:
            accumulated += chunk
            logger.info("Received %d bytes waiting for login prompt", len(chunk))
            if "login" in accumulated.lower():
                return accumulated
    raise AssertionError(
        f"Console did not show login prompt within {timeout:.0f}s. Received {len(accumulated)} bytes total."
    )


def _authenticate(
    *,
    send: Callable[[bytes], None],
    recv: Callable[[float], str | None],
    username: str,
    password: str,
    timeout: float = _AUTH_TIMEOUT_S,
) -> None:
    send(f"{username}\n".encode())
    accumulated = _collect_until(recv=recv, timeout=timeout, done=lambda text: "password" in text.lower())
    if "password" not in accumulated.lower():
        raise AssertionError(f"Console did not prompt for password. Received: {accumulated[-500:]!r}")
    send(f"{password}\n".encode())
    accumulated += _collect_until(recv=recv, timeout=timeout, done=lambda text: _logged_in(text, username))
    if not _logged_in(accumulated, username):
        raise AssertionError(f"Console login failed for {username}. Received: {accumulated[-500:]!r}")


def _logged_in(text: str, username: str) -> bool:
    lower = text.lower()
    if "login incorrect" in lower:
        return False
    return "$" in text or "#" in text or f"{username}@" in lower


def _run_ping(
    *,
    send: Callable[[bytes], None],
    recv: Callable[[float], str | None],
    dest_ip: str,
    timeout: float = _PING_TIMEOUT_S,
) -> bool:
    send(f"ping -c 3 -W 2 {dest_ip}; echo PING_RC:$?\n".encode())
    accumulated = _collect_until(recv=recv, timeout=timeout, done=lambda text: "PING_RC:" in text)
    logger.info("Guest ping output (%d bytes): %s", len(accumulated), accumulated[-1024:])
    match = re.search(r"PING_RC:(\d+)", accumulated)
    if match:
        return match.group(1) == "0"
    lower = accumulated.lower()
    if "100% packet loss" in lower or "network is unreachable" in lower:
        return False
    return bool(re.search(r"[1-9]\d* (packets )?received", lower)) and " 0% packet loss" in lower


def _collect_until(*, recv: Callable[[float], str | None], timeout: float, done: Callable[[str], bool]) -> str:
    accumulated = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        chunk = recv(min(remaining, 5.0))
        if chunk:
            accumulated += chunk
            if done(accumulated):
                return accumulated
    return accumulated
