# SPDX-License-Identifier: Apache-2.0
"""Single control-plane HTTP client factory (CL-05/CL-11).

Every outbound cluster request goes through this one place so the
security properties hold everywhere: redirects are never followed (a
redirect would replay the credential to a host the operator never named),
timeouts are always explicit, the credential travels in the Authorization
header and never in a URL, and the target host is validated before a
connection is attempted.

``tls_context`` is the TLS seam reserved by CL-05; v1 speaks plaintext
HTTP and passes None.
"""

from __future__ import annotations

import logging
import ssl
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from ..utils.network import is_valid_alias

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 15.0


class ClusterClientError(RuntimeError):
    """A control-plane request failed."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_head_url(url: str) -> str:
    """Normalize and validate an operator-supplied head URL.

    Returns the base URL without a trailing slash. Raises ValueError for
    anything that is not a plain http(s) URL naming a valid host.
    """
    candidate = (url or "").strip()
    if not candidate:
        raise ValueError("Head URL must not be empty")
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported head URL scheme: {parsed.scheme or '(none)'}")
    if parsed.username or parsed.password:
        raise ValueError("Head URL must not carry credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Head URL must not carry a query string or fragment")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Head URL has no host: {url}")
    if not is_valid_alias(hostname):
        raise ValueError(f"Invalid head URL host: {hostname}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid head URL port: {url}") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"Invalid head URL port: {port}")
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


class ClusterClient:
    """Minimal JSON client for worker → head control-plane calls."""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
        tls_context: ssl.SSLContext | None = None,
    ) -> None:
        self.base_url = validate_head_url(base_url)
        self._tls_context = tls_context
        self._timeout = httpx.Timeout(
            connect=connect_timeout_s,
            read=read_timeout_s,
            write=read_timeout_s,
            pool=connect_timeout_s,
        )

    def _build(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": self._timeout,
            "follow_redirects": False,
        }
        if self._tls_context is not None:
            kwargs["verify"] = self._tls_context
        return httpx.AsyncClient(**kwargs)

    async def post_json(
        self, path: str, *, token: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request("POST", path, token=token, payload=payload)

    async def get_json(self, path: str, *, token: str) -> dict[str, Any]:
        return await self._request("GET", path, token=token, payload=None)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with self._build() as client:
                response = await client.request(
                    method, path, headers=headers, json=payload
                )
        except httpx.HTTPError as exc:
            raise ClusterClientError(
                f"Control-plane request to {self.base_url}{path} failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ClusterClientError(
                f"{method} {self.base_url}{path} returned "
                f"{response.status_code}: {_detail(response)}",
                status_code=response.status_code,
            )
        if response.is_redirect:
            raise ClusterClientError(
                f"{method} {self.base_url}{path} was redirected; "
                "the control plane never follows redirects",
                status_code=response.status_code,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ClusterClientError(
                f"{method} {self.base_url}{path} returned a non-JSON body"
            ) from exc
        if not isinstance(body, dict):
            raise ClusterClientError(
                f"{method} {self.base_url}{path} returned an unexpected body type"
            )
        return body


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:200]
    return str(body)[:200]
