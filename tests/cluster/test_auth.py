# SPDX-License-Identifier: Apache-2.0
"""Tests for the cluster auth dependencies (E7 / CL-01 / CL-02).

Each dependency is exercised directly against a synthetic request so the
fail-closed behavior is asserted at the dependency, not only through a
route that happens to mount it.
"""

import pytest
from fastapi import HTTPException, Request

from omlx.admin.auth import create_session_token, fingerprint_key
from omlx.cluster.auth import (
    require_bootstrap_token,
    require_cluster_enabled,
    require_cluster_member,
    require_cluster_operator,
    require_cluster_operator_or_member,
    require_head_role,
    require_worker_role,
)
from omlx.cluster.credentials import generate_secret
from omlx.cluster.manager import set_cluster_manager
from omlx.cluster.versions import collect_versions

from .conftest import MAIN_API_KEY, SUB_KEY, make_settings, running_manager


def request_with(headers: dict[str, str] | None = None, cookies: str = "") -> Request:
    """Build a bare ASGI request carrying the given headers."""
    raw = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    if cookies:
        raw.append((b"cookie", cookies.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/cluster/state",
        "headers": raw,
        "query_string": b"",
        "client": ("10.0.0.9", 5000),
    }
    return Request(scope)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def admit(manager, host="10.0.0.9"):
    return await manager.join(
        peer_host=host, port=8000, name="w", versions=collect_versions().to_dict()
    )


class TestRoleGating:
    async def test_everything_404s_with_no_manager_installed(self):
        set_cluster_manager(None)
        for dependency in (
            require_cluster_enabled,
            require_head_role,
            require_worker_role,
        ):
            with pytest.raises(HTTPException) as exc:
                await dependency()
            assert exc.value.status_code == 404

    async def test_head_routes_404_on_a_worker(self, worker_settings):
        async with running_manager(worker_settings):
            with pytest.raises(HTTPException) as exc:
                await require_head_role()
            assert exc.value.status_code == 404

    async def test_worker_routes_404_on_a_head(self, head_settings):
        async with running_manager(head_settings):
            with pytest.raises(HTTPException) as exc:
                await require_worker_role()
            assert exc.value.status_code == 404


class TestMemberCredential:
    async def test_valid_member_secret_is_accepted(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = await require_cluster_member(
                request_with(auth(joined["member_secret"]))
            )
            assert member.id == joined["member_id"]

    async def test_missing_and_invalid_credentials_are_refused(self, head_settings):
        async with running_manager(head_settings):
            for headers in (
                {},
                auth(""),
                auth(generate_secret()),
                {"Authorization": "Basic x"},
            ):
                with pytest.raises(HTTPException) as exc:
                    await require_cluster_member(request_with(headers))
                assert exc.value.status_code == 401

    async def test_main_api_key_is_not_a_member_credential(self, head_settings):
        async with running_manager(head_settings):
            with pytest.raises(HTTPException):
                await require_cluster_member(request_with(auth(MAIN_API_KEY)))

    async def test_revoked_member_is_refused(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            await manager.remove_member(joined["member_id"])
            with pytest.raises(HTTPException) as exc:
                await require_cluster_member(
                    request_with(auth(joined["member_secret"]))
                )
            assert exc.value.status_code == 401


class TestOperatorCredential:
    async def test_main_key_is_accepted(self, head_settings):
        async with running_manager(head_settings):
            assert await require_cluster_operator(request_with(auth(MAIN_API_KEY)))

    async def test_admin_session_cookie_is_accepted(self, head_settings):
        async with running_manager(head_settings):
            cookie = f"omlx_admin_session={create_session_token()}"
            assert await require_cluster_operator(request_with(cookies=cookie))

    async def test_sub_key_is_refused(self, head_settings):
        """CL-02: a sub key is an inference credential, never an operator one."""
        async with running_manager(head_settings):
            with pytest.raises(HTTPException) as exc:
                await require_cluster_operator(request_with(auth(SUB_KEY)))
            assert exc.value.status_code == 401

    async def test_member_secret_is_refused(self, head_settings):
        """The privilege boundary is one-way."""
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            with pytest.raises(HTTPException) as exc:
                await require_cluster_operator(
                    request_with(auth(joined["member_secret"]))
                )
            assert exc.value.status_code == 401

    async def test_skip_api_key_verification_is_ignored(self, head_settings):
        """CL-01: the global bypass flag never reaches a cluster dependency."""
        async with running_manager(head_settings) as manager:
            manager.global_settings.auth.skip_api_key_verification = True
            with pytest.raises(HTTPException) as exc:
                await require_cluster_operator(request_with())
            assert exc.value.status_code == 401

    async def test_absent_api_key_does_not_allow_everything(self, head_settings):
        """CL-01: 'no key configured means allow' never reaches cluster auth."""
        async with running_manager(head_settings) as manager:
            manager.global_settings.auth.api_key = None
            for headers in ({}, auth(""), auth("anything")):
                with pytest.raises(HTTPException) as exc:
                    await require_cluster_operator(request_with(headers))
                assert exc.value.status_code == 401


class TestBootstrapCredential:
    async def test_valid_token_is_accepted(self, head_settings):
        async with running_manager(head_settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            assert await require_bootstrap_token(request_with(auth(token)))

    async def test_revoked_token_is_refused(self, head_settings):
        async with running_manager(head_settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            await manager.revoke_bootstrap_token()
            with pytest.raises(HTTPException) as exc:
                await require_bootstrap_token(request_with(auth(token)))
            assert exc.value.status_code == 401

    async def test_expired_token_is_refused(self, tmp_path):
        settings = make_settings(
            tmp_path / "ttl", role="head", bootstrap_token_ttl_s=-1.0
        )
        async with running_manager(settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            with pytest.raises(HTTPException) as exc:
                await require_bootstrap_token(request_with(auth(token)))
            assert exc.value.status_code == 401

    async def test_member_secret_is_not_a_bootstrap_token(self, head_settings):
        async with running_manager(head_settings) as manager:
            await manager.mint_bootstrap_token()
            joined = await admit(manager)
            with pytest.raises(HTTPException):
                await require_bootstrap_token(
                    request_with(auth(joined["member_secret"]))
                )

    async def test_missing_token_is_refused(self, head_settings):
        async with running_manager(head_settings) as manager:
            await manager.mint_bootstrap_token()
            with pytest.raises(HTTPException) as exc:
                await require_bootstrap_token(request_with())
            assert exc.value.status_code == 401


class TestRejectionLogging:
    """CL-07: a rejected credential is logged by fingerprint, never verbatim."""

    async def test_rejected_credentials_are_logged_by_fingerprint_only(
        self, head_settings, caplog
    ):
        presented = generate_secret()
        dependencies = (
            require_cluster_member,
            require_cluster_operator,
            require_bootstrap_token,
        )
        async with running_manager(head_settings) as manager:
            await manager.mint_bootstrap_token()
            for dependency in dependencies:
                caplog.clear()
                with (
                    caplog.at_level("WARNING", logger="omlx.cluster.auth"),
                    pytest.raises(HTTPException),
                ):
                    await dependency(request_with(auth(presented)))
                logged = "\n".join(record.getMessage() for record in caplog.records)
                assert logged, dependency.__name__
                assert presented not in logged, dependency.__name__
                assert fingerprint_key(presented) in logged, dependency.__name__


class TestReadTier:
    async def test_operator_may_read(self, head_settings):
        async with running_manager(head_settings):
            assert await require_cluster_operator_or_member(
                request_with(auth(MAIN_API_KEY))
            )

    async def test_member_may_read(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            assert await require_cluster_operator_or_member(
                request_with(auth(joined["member_secret"]))
            )

    async def test_sub_key_may_not_read(self, head_settings):
        async with running_manager(head_settings):
            with pytest.raises(HTTPException) as exc:
                await require_cluster_operator_or_member(request_with(auth(SUB_KEY)))
            assert exc.value.status_code == 401

    async def test_unauthenticated_read_is_refused(self, head_settings):
        async with running_manager(head_settings):
            with pytest.raises(HTTPException) as exc:
                await require_cluster_operator_or_member(request_with())
            assert exc.value.status_code == 401
