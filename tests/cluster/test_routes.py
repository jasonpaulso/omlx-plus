# SPDX-License-Identifier: Apache-2.0
"""Tests for the /v1/cluster/* router: role gating, tiers, and payloads."""

import pytest

from omlx.cluster.credentials import generate_secret
from omlx.cluster.versions import PackageVersion, VersionInfo, collect_versions

from .conftest import (
    MAIN_API_KEY,
    SUB_KEY,
    bearer,
    build_app,
    http_client,
    make_settings,
    running_manager,
)

HEAD_ROUTES = [
    ("GET", "/v1/cluster/state", "read"),
    ("POST", "/v1/cluster/token", "operator"),
    ("DELETE", "/v1/cluster/token", "operator"),
    ("DELETE", "/v1/cluster/members/mystery", "operator"),
    ("POST", "/v1/cluster/join", "bootstrap"),
    ("POST", "/v1/cluster/heartbeat", "member"),
    ("POST", "/v1/cluster/leave", "member"),
]
LOCAL_ROUTES = [
    ("POST", "/v1/cluster/local/join"),
    ("POST", "/v1/cluster/local/leave"),
    ("GET", "/v1/cluster/local/status"),
]
ALLOWED = {
    "read": {"operator", "member"},
    "operator": {"operator"},
    "bootstrap": {"bootstrap"},
    "member": {"member"},
}


async def admit(manager, host="10.0.0.9"):
    return await manager.join(
        peer_host=host, port=8000, name="w", versions=collect_versions().to_dict()
    )


class TestRoleGating:
    async def test_every_cluster_route_404s_when_the_role_is_off(self, tmp_path):
        """Flag off is indistinguishable from a build without the feature."""
        settings = make_settings(tmp_path / "off", role="off")
        async with running_manager(settings):
            app = build_app()
            async with http_client(app) as client:
                for method, path, _tier in HEAD_ROUTES:
                    assert (await client.request(method, path)).status_code == 404
                    response = await client.request(
                        method, path, headers=bearer(MAIN_API_KEY)
                    )
                    assert response.status_code == 404
                for method, path in LOCAL_ROUTES:
                    assert (await client.request(method, path)).status_code == 404

    async def test_head_routes_404_on_a_worker_even_for_an_operator(
        self, worker_settings
    ):
        async with running_manager(worker_settings):
            app = build_app()
            async with http_client(app) as client:
                for method, path, _tier in HEAD_ROUTES:
                    response = await client.request(
                        method, path, headers=bearer(MAIN_API_KEY), json={}
                    )
                    assert response.status_code == 404, (method, path)

    async def test_local_routes_404_on_a_head_even_for_an_operator(self, head_settings):
        async with running_manager(head_settings):
            app = build_app()
            async with http_client(app) as client:
                for method, path in LOCAL_ROUTES:
                    response = await client.request(
                        method, path, headers=bearer(MAIN_API_KEY), json={}
                    )
                    assert response.status_code == 404, (method, path)

    async def test_role_is_checked_before_credentials(self, worker_settings):
        """A wrong-role request 404s rather than leaking a 401 distinction."""
        async with running_manager(worker_settings):
            app = build_app()
            async with http_client(app) as client:
                response = await client.post(
                    "/v1/cluster/token", headers=bearer("garbage")
                )
                assert response.status_code == 404


class TestTierMatrix:
    @pytest.mark.parametrize("method,path,tier", HEAD_ROUTES)
    async def test_only_the_routes_own_tier_is_accepted(
        self, head_settings, method, path, tier
    ):
        async with running_manager(head_settings) as manager:
            bootstrap = (await manager.mint_bootstrap_token())["token"]
            member_secret = (await admit(manager))["member_secret"]
            credentials = {
                "operator": MAIN_API_KEY,
                "member": member_secret,
                "bootstrap": bootstrap,
                "sub_key": SUB_KEY,
                "unknown": generate_secret(),
            }
            app = build_app()
            async with http_client(app) as client:
                for name, token in credentials.items():
                    response = await client.request(
                        method, path, headers=bearer(token), json={}
                    )
                    if name in ALLOWED[tier]:
                        assert response.status_code != 401, (path, name)
                    else:
                        assert response.status_code == 401, (path, name)

                unauthenticated = await client.request(method, path, json={})
                assert unauthenticated.status_code == 401

    @pytest.mark.parametrize("method,path", LOCAL_ROUTES)
    async def test_local_routes_are_operator_only(self, worker_settings, method, path):
        async with running_manager(worker_settings):
            app = build_app()
            async with http_client(app) as client:
                for token in (SUB_KEY, generate_secret(), ""):
                    response = await client.request(
                        method, path, headers=bearer(token), json={}
                    )
                    assert response.status_code == 401
                allowed = await client.request(
                    method, path, headers=bearer(MAIN_API_KEY), json={}
                )
                assert allowed.status_code != 401


class TestBypassFlagsNeverApply:
    @pytest.mark.parametrize(
        "role,method,path",
        [
            ("worker", "POST", "/v1/cluster/local/join"),
            ("head", "POST", "/v1/cluster/token"),
        ],
    )
    async def test_skip_flag_and_absent_api_key_do_not_open_the_endpoint(
        self, tmp_path, role, method, path
    ):
        """CL-01, asserted on the specifically named endpoints."""
        settings = make_settings(tmp_path / role, role=role)
        async with running_manager(settings) as manager:
            manager.global_settings.auth.skip_api_key_verification = True
            manager.global_settings.auth.api_key = None
            app = build_app()
            async with http_client(app) as client:
                for headers in ({}, bearer(""), bearer("anything")):
                    response = await client.request(
                        method, path, headers=headers, json={}
                    )
                    assert response.status_code == 401

    async def test_admin_cluster_endpoint_is_not_opened_by_the_skip_flag(
        self, head_settings
    ):
        """CL-01 for GET /admin/api/cluster (not behind require_admin)."""
        async with running_manager(head_settings) as manager:
            manager.global_settings.auth.skip_api_key_verification = True
            manager.global_settings.auth.api_key = None
            app = build_app(with_admin=True)
            async with http_client(app) as client:
                assert (await client.get("/admin/api/cluster")).status_code == 401
                response = await client.get(
                    "/admin/api/cluster", headers=bearer(SUB_KEY)
                )
                assert response.status_code == 401

    async def test_admin_cluster_endpoint_serves_operators(self, head_settings):
        async with running_manager(head_settings) as manager:
            await admit(manager)
            app = build_app(with_admin=True)
            async with http_client(app) as client:
                response = await client.get(
                    "/admin/api/cluster", headers=bearer(MAIN_API_KEY)
                )
            assert response.status_code == 200
            assert response.json()["member_count"] == 1

    async def test_setup_api_key_is_inert_once_a_key_is_configured(
        self, head_settings, monkeypatch
    ):
        """The session-mint path is inert in every reachable cluster state."""
        import omlx.server as server_module

        # The route resolves settings through the server state, so patch
        # there: importing omlx.server re-registers the admin getters and
        # would undo a patch of the getter itself.
        monkeypatch.setattr(
            server_module._server_state, "global_settings", head_settings
        )
        async with running_manager(head_settings):
            app = build_app(with_admin=True)
            async with http_client(app) as client:
                response = await client.post(
                    "/admin/api/setup-api-key",
                    json={"api_key": "new-key-1234", "api_key_confirm": "new-key-1234"},
                )
            assert response.status_code == 400
            assert "already configured" in response.json()["detail"]


class TestClusterCredentialStaysInsideTheClusterSurface:
    async def test_cluster_credential_is_refused_on_inference_and_admin_routes(
        self, head_settings
    ):
        """CL-02: the privilege boundary is one-way."""
        import omlx.server as server_module

        async with running_manager(head_settings) as manager:
            secret = (await admit(manager))["member_secret"]
            bootstrap = (await manager.mint_bootstrap_token())["token"]

        state = server_module._server_state
        previous = (state.api_key, state.global_settings)
        state.api_key = MAIN_API_KEY
        state.global_settings = head_settings
        try:
            async with http_client(server_module.app) as client:
                for token in (secret, bootstrap):
                    inference = await client.post(
                        "/v1/chat/completions",
                        headers=bearer(token),
                        json={"model": "m", "messages": []},
                    )
                    assert inference.status_code == 401
                    admin = await client.get(
                        "/admin/api/global-settings", headers=bearer(token)
                    )
                    assert admin.status_code == 401
        finally:
            state.api_key, state.global_settings = previous


class TestJoinEndpoint:
    async def test_address_comes_from_the_socket_not_the_body(self, head_settings):
        """CL-10: a peer cannot choose the address recorded for it."""
        async with running_manager(head_settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            app = build_app()
            async with http_client(app, peer=("10.9.9.9", 33333)) as client:
                response = await client.post(
                    "/v1/cluster/join",
                    headers=bearer(token),
                    json={
                        "port": 8000,
                        "address": "192.168.0.1",
                        "versions": collect_versions().to_dict(),
                    },
                )
            assert response.status_code == 200
            member = manager.state.member(response.json()["member_id"])
            assert str(member.address) == "10.9.9.9"

    async def test_loopback_peer_is_rejected_without_test_mode(self, tmp_path):
        settings = make_settings(tmp_path / "strict", role="head", allow_loopback=False)
        async with running_manager(settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            app = build_app()
            async with http_client(app, peer=("127.0.0.1", 5000)) as client:
                response = await client.post(
                    "/v1/cluster/join",
                    headers=bearer(token),
                    json={"port": 8000, "versions": collect_versions().to_dict()},
                )
            assert response.status_code == 400

    async def test_version_skew_is_rejected_over_http(self, head_settings):
        async with running_manager(head_settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            skewed = VersionInfo(
                omlx="9.9.9", mlx="9.9.9", mlx_lm=PackageVersion("9.9.9", "beefbee")
            )
            app = build_app()
            async with http_client(app) as client:
                response = await client.post(
                    "/v1/cluster/join",
                    headers=bearer(token),
                    json={"port": 8000, "versions": skewed.to_dict()},
                )
            assert response.status_code == 409
            detail = response.json()["detail"]
            assert "9.9.9" in detail and manager.versions.omlx in detail

    async def test_out_of_range_port_is_refused(self, head_settings):
        async with running_manager(head_settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            app = build_app()
            async with http_client(app) as client:
                response = await client.post(
                    "/v1/cluster/join",
                    headers=bearer(token),
                    json={"port": 70000, "versions": collect_versions().to_dict()},
                )
            assert response.status_code == 422


class TestHeartbeatAndLeaveEndpoints:
    async def test_heartbeat_then_state_shows_the_member_active(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            app = build_app()
            async with http_client(app) as client:
                beat = await client.post(
                    "/v1/cluster/heartbeat",
                    headers=bearer(joined["member_secret"]),
                    json={"seq": 1, "epoch": "e1"},
                )
                assert beat.status_code == 200
                state = await client.get(
                    "/v1/cluster/state", headers=bearer(MAIN_API_KEY)
                )
            member = state.json()["members"][0]
            assert member["status"] == "active"
            assert member["last_seq"] == 1

    async def test_replayed_heartbeat_is_refused(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            app = build_app()
            headers = bearer(joined["member_secret"])
            async with http_client(app) as client:
                await client.post(
                    "/v1/cluster/heartbeat",
                    headers=headers,
                    json={"seq": 4, "epoch": "e1"},
                )
                replay = await client.post(
                    "/v1/cluster/heartbeat",
                    headers=headers,
                    json={"seq": 4, "epoch": "e1"},
                )
            assert replay.status_code == 409

    async def test_after_leave_the_secret_no_longer_authenticates(self, head_settings):
        """CL-03: revocation is real and scoped to the member that left."""
        async with running_manager(head_settings) as manager:
            first = await admit(manager, host="10.0.0.9")
            second = await admit(manager, host="10.0.0.10")
            app = build_app()
            async with http_client(app) as client:
                left = await client.post(
                    "/v1/cluster/leave", headers=bearer(first["member_secret"])
                )
                assert left.status_code == 200

                gone = await client.post(
                    "/v1/cluster/heartbeat",
                    headers=bearer(first["member_secret"]),
                    json={"seq": 1, "epoch": "e1"},
                )
                assert gone.status_code == 401

                survivor = await client.post(
                    "/v1/cluster/heartbeat",
                    headers=bearer(second["member_secret"]),
                    json={"seq": 1, "epoch": "e1"},
                )
                assert survivor.status_code == 200

    async def test_operator_removal_revokes_the_member(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            app = build_app()
            async with http_client(app) as client:
                removed = await client.delete(
                    f"/v1/cluster/members/{joined['member_id']}",
                    headers=bearer(MAIN_API_KEY),
                )
                assert removed.status_code == 200
                beat = await client.post(
                    "/v1/cluster/heartbeat",
                    headers=bearer(joined["member_secret"]),
                    json={"seq": 1, "epoch": "e1"},
                )
            assert beat.status_code == 401

    async def test_removing_an_unknown_member_is_404(self, head_settings):
        async with running_manager(head_settings):
            app = build_app()
            async with http_client(app) as client:
                response = await client.delete(
                    "/v1/cluster/members/nope", headers=bearer(MAIN_API_KEY)
                )
            assert response.status_code == 404


class TestStateEndpoint:
    async def test_state_carries_no_credential_material(self, head_settings):
        """CL-12: the read endpoint never returns secrets or digests."""
        async with running_manager(head_settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            joined = await admit(manager)
            app = build_app()
            async with http_client(app) as client:
                response = await client.get(
                    "/v1/cluster/state", headers=bearer(MAIN_API_KEY)
                )
            body = response.text
            assert token not in body
            assert joined["member_secret"] not in body
            assert manager.state.bootstrap.digest not in body
            for digest in manager.state.member_digests.values():
                assert digest not in body

    async def test_a_member_may_read_state(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            app = build_app()
            async with http_client(app) as client:
                response = await client.get(
                    "/v1/cluster/state", headers=bearer(joined["member_secret"])
                )
            assert response.status_code == 200
            assert response.json()["members"][0]["id"] == joined["member_id"]

    async def test_get_requests_change_no_state(self, head_settings):
        async with running_manager(head_settings) as manager:
            await admit(manager)
            before = manager.state
            app = build_app()
            async with http_client(app) as client:
                await client.get("/v1/cluster/state", headers=bearer(MAIN_API_KEY))
            assert manager.state == before


class TestTokenEndpoint:
    async def test_mint_returns_the_value_once(self, head_settings):
        async with running_manager(head_settings) as manager:
            app = build_app()
            async with http_client(app) as client:
                response = await client.post(
                    "/v1/cluster/token", headers=bearer(MAIN_API_KEY)
                )
                assert response.status_code == 200
                token = response.json()["token"]

                state = await client.get(
                    "/v1/cluster/state", headers=bearer(MAIN_API_KEY)
                )
            assert token not in state.text
            assert state.json()["bootstrap_token"]["configured"] is True
            assert manager.state.bootstrap.digest != token

    async def test_revoked_token_no_longer_admits(self, head_settings):
        async with running_manager(head_settings):
            app = build_app()
            async with http_client(app) as client:
                token = (
                    await client.post("/v1/cluster/token", headers=bearer(MAIN_API_KEY))
                ).json()["token"]
                await client.delete("/v1/cluster/token", headers=bearer(MAIN_API_KEY))

                response = await client.post(
                    "/v1/cluster/join",
                    headers=bearer(token),
                    json={"port": 8000, "versions": collect_versions().to_dict()},
                )
            assert response.status_code == 401
