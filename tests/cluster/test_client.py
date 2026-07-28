# SPDX-License-Identifier: Apache-2.0
"""Tests for the control-plane HTTP client (CL-05/CL-11)."""

import httpx
import pytest

from omlx.cluster.client import ClusterClient, ClusterClientError, validate_head_url


class TestValidateHeadUrl:
    def test_bare_host_and_port_gets_http_scheme(self):
        assert validate_head_url("head.local:8000") == "http://head.local:8000"

    def test_trailing_slash_is_stripped(self):
        assert validate_head_url("http://head:8000/") == "http://head:8000"

    def test_https_is_accepted(self):
        assert validate_head_url("https://head:8443") == "https://head:8443"

    def test_ipv4_and_loopback_hosts_are_accepted(self):
        assert validate_head_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
        assert validate_head_url("http://10.0.0.4:8000") == "http://10.0.0.4:8000"

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "   ",
            "ftp://head:8000",
            "file:///etc/passwd",
            "http://999.999.999.999:8000",
            "http://:8000",
        ],
    )
    def test_rejects_unusable_urls(self, url):
        with pytest.raises(ValueError):
            validate_head_url(url)

    def test_rejects_embedded_credentials(self):
        with pytest.raises(ValueError, match="credentials"):
            validate_head_url("http://user:pass@head:8000")

    def test_rejects_query_and_fragment(self):
        """A token must never be smuggled into a URL (anti-pattern CL-12)."""
        with pytest.raises(ValueError, match="query string"):
            validate_head_url("http://head:8000/?token=secret")

    def test_rejects_unspecified_address(self):
        with pytest.raises(ValueError):
            validate_head_url("http://0.0.0.0:8000")


class TestClientConfiguration:
    def test_redirects_are_never_followed(self):
        client = ClusterClient("http://head:8000")
        assert client._build().follow_redirects is False

    def test_timeouts_are_explicit(self):
        client = ClusterClient(
            "http://head:8000", connect_timeout_s=1.5, read_timeout_s=2.5
        )
        timeout = client._build().timeout
        assert timeout.connect == 1.5
        assert timeout.read == 2.5

    def test_invalid_url_fails_at_construction(self):
        with pytest.raises(ValueError):
            ClusterClient("gopher://head")


def _mock_client(client: ClusterClient, handler):
    def _build() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=client.base_url,
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
        )

    return _build


class TestRequests:
    async def test_credential_travels_in_the_authorization_header(self):
        seen: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            return httpx.Response(200, json={"ok": True})

        client = ClusterClient("http://head:8000")
        client._build = _mock_client(client, handler)

        assert await client.post_json(
            "/v1/cluster/join", token="s3cret", payload={}
        ) == {"ok": True}
        request = seen["request"]
        assert request.headers["Authorization"] == "Bearer s3cret"
        assert "s3cret" not in str(request.url)

    async def test_redirect_is_reported_not_followed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(307, headers={"location": "http://evil/steal"})

        client = ClusterClient("http://head:8000")
        client._build = _mock_client(client, handler)

        with pytest.raises(ClusterClientError, match="redirected"):
            await client.post_json("/v1/cluster/join", token="s3cret", payload={})

    async def test_error_status_is_raised_with_detail(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "Invalid token"})

        client = ClusterClient("http://head:8000")
        client._build = _mock_client(client, handler)

        with pytest.raises(ClusterClientError) as exc:
            await client.post_json("/v1/cluster/join", token="bad", payload={})
        assert exc.value.status_code == 401
        assert "Invalid token" in str(exc.value)

    async def test_transport_failure_is_wrapped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        client = ClusterClient("http://head:8000")
        client._build = _mock_client(client, handler)

        with pytest.raises(ClusterClientError, match="failed"):
            await client.get_json("/v1/cluster/state", token="tok")

    async def test_non_json_body_is_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        client = ClusterClient("http://head:8000")
        client._build = _mock_client(client, handler)

        with pytest.raises(ClusterClientError, match="non-JSON"):
            await client.get_json("/v1/cluster/state", token="tok")
