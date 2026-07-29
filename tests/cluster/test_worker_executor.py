# SPDX-License-Identifier: Apache-2.0
"""Worker-side command confinement (CL2-01 … CL2-12).

A compromised or impersonated head must never become code execution on the
worker: every command is untrusted input, confined against the worker's OWN
settings. These are the P2 CL2 acceptance deltas from the plan's Tests section.
"""

from __future__ import annotations

import asyncio
import time

from omlx.cluster import hostfile
from omlx.cluster.manager import WorkerCommandExecutor
from omlx.cluster.protocol import PROTOCOL_VERSION, SpawnRankCommand

from .conftest import make_settings


class FakeCluster:
    def __init__(self) -> None:
        self._alive = True

    def any_alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self._alive = False


def _executor(tmp_path, spawns, *, present=True, **cluster):
    opts = {"data_plane_subnet": "10.0.2.0/24", "data_plane_address": "10.0.2.2"}
    opts.update(cluster)
    settings = make_settings(tmp_path, role="worker", **opts)

    def spawn(prepared):
        cluster_obj = FakeCluster()
        spawns.append(prepared)
        return cluster_obj

    def resolve(model_id):
        if present and model_id == "target-model":
            return ("/models/target", 1234)
        return None

    return WorkerCommandExecutor(
        settings,
        spawn_fn=spawn,
        model_resolver=resolve,
        local_addresses={"10.0.2.2"},
    )


def _spawn_cmd(**over):
    base = {
        "kind": "spawn_rank",
        "schema_version": PROTOCOL_VERSION,
        "job_id": "j1",
        "step": 1,
        "rank": 1,
        "world_size": 2,
        "backend": "ring",
        "model_id": "target-model",
        "peers": ["10.0.2.1", "10.0.2.9"],
        "base_port": 41100,
    }
    base.update(over)
    return base


# -- CL2-03: own rank computed locally, every entry re-validated --------------


async def test_valid_spawn_uses_own_address_for_own_rank(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns)
    update = await executor._apply(_spawn_cmd())
    assert update["status"] == "spawned"
    assert len(spawns) == 1
    prepared = spawns[0]
    # The head sent 10.0.2.9 for the worker's own rank; the worker ignored it
    # and used its own configured address.
    assert prepared.ips == ["10.0.2.1", "10.0.2.2"]
    assert prepared.model_path == "/models/target"
    assert prepared.base_port == 41100


async def test_peer_outside_own_subnet_is_refused(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns)
    update = await executor._apply(_spawn_cmd(peers=["10.0.9.9", "10.0.2.2"]))
    assert update["status"] == "error"
    assert not spawns


# -- jaccl: CL2-03 own row from own device; CL2-12 own device required --------


async def test_jaccl_spawn_overwrites_own_row_with_own_device(tmp_path):
    # The head supplies a matrix whose row for the worker's own rank names a
    # different device; the worker ignores it and uses its own rdma_device, and
    # keeps the head's peer rows (devices it cannot itself observe). CL2-03 for
    # the ibv matrix, exactly as `peers` is confined for addresses.
    spawns: list = []
    executor = _executor(tmp_path, spawns, rdma_device="rdma_en4")
    head_matrix = [[None, "rdma_en2"], ["rdma_head_lies", None]]
    update = await executor._apply(_spawn_cmd(backend="jaccl", ibv_devices=head_matrix))
    assert update["status"] == "spawned"
    prepared = spawns[0]
    assert prepared.backend == "jaccl"
    # Own row (rank 1) recomputed from own device; head's row 0 kept as sent.
    assert prepared.ibv_devices == [[None, "rdma_en2"], ["rdma_en4", None]]


async def test_jaccl_without_own_rdma_device_refuses(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns, rdma_device="")
    head_matrix = [[None, "rdma_en2"], ["rdma_en4", None]]
    update = await executor._apply(_spawn_cmd(backend="jaccl", ibv_devices=head_matrix))
    assert update["status"] == "error"
    assert "CL2-12" in update["detail"]
    assert not spawns


async def test_jaccl_without_a_matrix_refuses(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns, rdma_device="rdma_en4")
    update = await executor._apply(_spawn_cmd(backend="jaccl"))  # no ibv_devices
    assert update["status"] == "error"
    assert not spawns


def test_worker_resolves_both_backends(tmp_path):
    from omlx.cluster.protocol import Backend

    executor = _executor(tmp_path, [], rdma_device="rdma_en4")
    assert executor._resolve_backend(Backend.RING) == "ring"
    assert executor._resolve_backend(Backend.JACCL) == "jaccl"


async def test_presence_reports_own_rdma_device(tmp_path):
    executor = _executor(tmp_path, [], rdma_device="rdma_en4")
    update = await executor._apply(
        {
            "kind": "presence",
            "schema_version": PROTOCOL_VERSION,
            "job_id": "j",
            "step": 1,
            "model_id": "target-model",
        }
    )
    assert update["rdma_device"] == "rdma_en4"


# -- CL2-12: no own data-plane config refuses to form -------------------------


async def test_no_dataplane_config_refuses(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns, data_plane_subnet="")
    update = await executor._apply(_spawn_cmd())
    assert update["status"] == "error"
    assert "CL2-12" in update["detail"]
    assert not spawns


# -- CL2-02: model resolved against own dirs, never a path --------------------


async def test_absent_model_named_error_no_spawn(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns, present=False)
    update = await executor._apply(_spawn_cmd())
    assert update["status"] == "error"
    assert "not present" in update["detail"]
    assert not spawns


def test_no_command_shape_can_carry_a_path():
    # The spawn command is a closed schema with a model IDENTIFIER only; there
    # is no field that could carry a filesystem path.
    fields = set(SpawnRankCommand.model_fields)
    assert "model_id" in fields
    assert not fields & {"path", "model_path", "cwd", "dst"}


# -- CL2-04: fail closed on anything unexpected -------------------------------


async def test_unknown_kind_rejected_nothing_spawns(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns)
    update = await executor._apply(
        {"kind": "nope", "schema_version": PROTOCOL_VERSION, "job_id": "j", "step": 1}
    )
    assert update["status"] == "rejected"
    assert not spawns


async def test_off_version_command_rejected(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns)
    update = await executor._apply(_spawn_cmd(schema_version=PROTOCOL_VERSION + 1))
    assert update["status"] == "rejected"
    assert not spawns


async def test_unknown_field_rejected(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns)
    update = await executor._apply(_spawn_cmd(PYTHONPATH="/evil"))
    assert update["status"] == "rejected"
    assert not spawns


# -- CL2-01: no env crosses the wire; the rank env is built locally -----------


def test_local_env_drops_wire_shaped_keys():
    # Even if an attacker could smuggle env-shaped keys into a base env, the
    # allowlist local builder keeps only allowlisted keys and overlays the
    # locally-computed topology — an injected MLX_RANK/PYTHONPATH cannot survive.
    poisoned = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/evil",
        "DYLD_INSERT_LIBRARIES": "/evil.dylib",
        "MLX_RANK": "999",
    }
    env = hostfile.local_worker_env(
        poisoned, rank=1, backend="ring", hostfile="/tmp/hosts.json"
    )
    assert "PYTHONPATH" not in env
    assert "DYLD_INSERT_LIBRARIES" not in env
    assert env["MLX_RANK"] == "1"
    assert env["PATH"] == "/usr/bin"


# -- CL2-09: one live formation per worker ------------------------------------


async def test_second_formation_refused_while_one_is_live(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns)
    first = await executor._apply(_spawn_cmd(job_id="a", step=1))
    assert first["status"] == "spawned"
    second = await executor._apply(_spawn_cmd(job_id="b", step=1))
    assert second["status"] == "error"
    assert len(spawns) == 1


# -- CL2-06: a replayed command spawns nothing a second time ------------------


async def test_replayed_command_causes_no_second_spawn(tmp_path):
    spawns: list = []
    executor = _executor(tmp_path, spawns)
    await executor.start()
    try:
        executor.deliver([_spawn_cmd()])
        await _wait_for_updates(executor, 1)
        executor.deliver([_spawn_cmd()])  # same (job_id, step)
        await asyncio.sleep(0.05)
        assert len(spawns) == 1
    finally:
        await executor.stop()


async def _wait_for_updates(executor, count, timeout=2.0):
    deadline = time.monotonic() + timeout
    collected: list = []
    while time.monotonic() < deadline:
        collected += executor.pending_job_updates()
        if len(collected) >= count:
            return collected
        await asyncio.sleep(0.01)
    return collected


# -- presence resolves id-only and reports the worker's own address -----------


async def test_presence_answers_present_and_reports_own_address(tmp_path):
    executor = _executor(tmp_path, [])
    update = await executor._apply(
        {
            "kind": "presence",
            "schema_version": PROTOCOL_VERSION,
            "job_id": "j",
            "step": 1,
            "model_id": "target-model",
        }
    )
    assert update["present"] is True
    assert update["data_plane_address"] == "10.0.2.2"


async def test_presence_answers_absent(tmp_path):
    executor = _executor(tmp_path, [], present=False)
    update = await executor._apply(
        {
            "kind": "presence",
            "schema_version": PROTOCOL_VERSION,
            "job_id": "j",
            "step": 1,
            "model_id": "target-model",
        }
    )
    assert update["present"] is False
