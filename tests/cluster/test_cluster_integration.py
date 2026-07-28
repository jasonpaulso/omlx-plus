# SPDX-License-Identifier: Apache-2.0
"""Two-node cluster integration test.

Spawns two real ``omlx serve`` processes on localhost, each with its own
base path and port, and drives the whole S1 lifecycle through the HTTP API
and the CLI verbs: mint token, join, heartbeat, SIGKILL the worker, watch
the scrub mark it lost, restart the worker and watch it revive, leave, and
force-remove.

Every wait has a deadline and every spawned process is killed by process
group in teardown, so a hang in one step cannot hang the suite.

Double-marked ``cluster`` and ``integration`` so the default CI selection
(``-m "not slow and not integration"``) keeps excluding it.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests

pytestmark = [pytest.mark.cluster, pytest.mark.integration]

API_KEY = "cluster-integration-key"
HEARTBEAT_INTERVAL_S = 1.0
MEMBER_TIMEOUT_S = 3.0
STARTUP_TIMEOUT_S = 120.0
POLL_TIMEOUT_S = 30.0
CLI_TIMEOUT_S = 60.0
SHUTDOWN_TIMEOUT_S = 20.0
TOKEN_RE = re.compile(r"\b[0-9a-f]{64}\b")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def child_env(base_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    # The developer shell may export OMLX_API_KEY; the node must use the
    # key from its own settings file.
    env.pop("OMLX_API_KEY", None)
    env["OMLX_BASE_PATH"] = str(base_path)
    env["PYTHONUNBUFFERED"] = "1"
    return env


@dataclass
class Node:
    """One spawned oMLX server with its own state directory."""

    name: str
    base_path: Path
    port: int
    role: str
    processes: list[subprocess.Popen] = field(default_factory=list)
    log_handles: list = field(default_factory=list)
    log: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        self.log = self.base_path / "spawn.log"
        self.write_settings()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def cluster_file(self) -> Path:
        return self.base_path / "cluster.json"

    @property
    def process(self) -> subprocess.Popen:
        return self.processes[-1]

    def write_settings(self) -> None:
        settings = {
            "version": "1.0",
            "server": {"host": "127.0.0.1", "port": self.port, "log_level": "info"},
            # No models and no HF cache discovery: the control plane needs no
            # engine, and scanning the developer's HF cache would only slow
            # startup down.
            "model": {"model_dirs": [str(self.base_path / "models")]},
            "huggingface": {"hf_cache_enabled": False},
            "auth": {"api_key": API_KEY},
            "cluster": {
                "role": self.role,
                "heartbeat_interval_s": HEARTBEAT_INTERVAL_S,
                "member_timeout_s": MEMBER_TIMEOUT_S,
                "bootstrap_token_ttl_s": 300.0,
                "allow_loopback": True,
                "node_name": self.name,
            },
        }
        (self.base_path / "settings.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )

    def start(self) -> None:
        # The handle has to outlive this call: the child writes to it for as
        # long as it runs. Closed in kill_all().
        handle = open(self.log, "ab")  # noqa: SIM115
        self.log_handles.append(handle)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omlx.cli",
                "serve",
                "--base-path",
                str(self.base_path),
            ],
            env=child_env(self.base_path),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        self.processes.append(process)
        self.wait_healthy(process)

    def wait_healthy(self, process: subprocess.Popen) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(
                    f"{self.name} exited early (rc={process.returncode}):\n"
                    f"{self.tail()}"
                )
            try:
                # Any answer means the port is serving; /health reports 503
                # while models load, which is not a failure here.
                requests.get(f"{self.url}/health", timeout=2)
                return
            except requests.RequestException:
                time.sleep(0.5)
        raise AssertionError(f"{self.name} never became reachable:\n{self.tail()}")

    def kill(
        self, process: subprocess.Popen | None = None, *, sig=signal.SIGKILL
    ) -> None:
        process = process or self.processes[-1]
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            process.kill()
        try:
            process.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=SHUTDOWN_TIMEOUT_S)

    def kill_all(self) -> None:
        for process in self.processes:
            self.kill(process)
        for handle in self.log_handles:
            handle.close()

    def tail(self, limit: int = 4000) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return "(no log)"

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        return requests.request(
            method,
            f"{self.url}{path}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
            **kwargs,
        )

    def cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "omlx.cli", *args],
            env=child_env(self.base_path),
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_S,
            cwd=str(Path(__file__).resolve().parents[2]),
        )


def wait_until(predicate, *, timeout: float = POLL_TIMEOUT_S, message: str = ""):
    """Poll until the predicate returns a truthy value or the deadline passes."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {message} (last value: {last!r})")


@pytest.fixture
def nodes(tmp_path):
    head = Node("head", tmp_path / "head", free_port(), "head")
    worker = Node("worker", tmp_path / "worker", free_port(), "worker")
    try:
        yield head, worker
    finally:
        worker.kill_all()
        head.kill_all()


def member_of(head: Node) -> dict:
    state = head.request("GET", "/v1/cluster/state")
    assert state.status_code == 200, state.text
    members = state.json()["members"]
    return members[0] if members else {}


def test_two_node_cluster_lifecycle(nodes):
    head, worker = nodes
    assert head.port != worker.port
    assert head.cluster_file != worker.cluster_file

    head.start()
    worker.start()

    # Role gating: each node only serves its own half of the API.
    assert head.request("GET", "/v1/cluster/local/status").status_code == 404
    assert worker.request("POST", "/v1/cluster/token").status_code == 404
    assert head.request("GET", "/v1/cluster/state").json()["members"] == []

    # Mint a bootstrap token through the CLI and join with it.
    minted = head.cli("cluster", "token")
    assert minted.returncode == 0, minted.stdout + minted.stderr
    token = TOKEN_RE.search(minted.stdout)
    assert token, minted.stdout
    token_file = worker.base_path / "join-token"
    token_file.write_text(token.group(0), encoding="utf-8")

    joined = worker.cli("join", head.url, "--token-file", str(token_file))
    assert joined.returncode == 0, joined.stdout + joined.stderr

    member = wait_until(
        lambda: member_of(head) or None, message="the worker to appear on the head"
    )
    member_id = member["id"]
    assert member["status"] == "active"
    assert member["address"] == "127.0.0.1"
    assert member["port"] == worker.port

    # Per-process state isolation: the worker holds its own credential and
    # no membership table; the head holds digests and no plaintext secret.
    worker_state = json.loads(worker.cluster_file.read_text(encoding="utf-8"))
    head_state = json.loads(head.cluster_file.read_text(encoding="utf-8"))
    secret = worker_state["worker"]["secret"]
    assert worker_state["members"] == []
    assert worker_state["worker"]["member_id"] == member_id
    assert head_state["worker"] is None
    assert secret not in json.dumps(head_state)
    assert member_id in head_state["member_digests"]
    assert oct(worker.cluster_file.stat().st_mode)[-3:] == "600"
    assert oct(head.cluster_file.stat().st_mode)[-3:] == "600"

    # Heartbeats keep arriving.
    first_beat = member_of(head)["last_heartbeat_at"]
    wait_until(
        lambda: (member_of(head).get("last_heartbeat_at") or 0) > first_beat,
        message="a further heartbeat",
    )

    # A hard kill leaves the head to notice by timeout.
    worker.kill()
    wait_until(
        lambda: member_of(head).get("status") == "lost",
        message="the scrub loop to mark the member lost",
    )
    # A timeout is not a revocation: the member is still admitted.
    assert member_of(head)["id"] == member_id

    # The worker restarts, resumes heartbeats from its persisted credential,
    # and revives with a new epoch.
    lost_epoch = member_of(head).get("epoch")
    worker.start()
    wait_until(
        lambda: member_of(head).get("status") == "active",
        message="the restarted worker to revive",
    )
    assert member_of(head)["epoch"] != lost_epoch

    status = worker.cli("cluster", "status")
    assert status.returncode == 0, status.stdout + status.stderr
    assert member_id in status.stdout
    assert secret not in status.stdout

    # Leaving revokes the credential for real.
    left = worker.cli("cluster", "leave")
    assert left.returncode == 0, left.stdout + left.stderr
    assert member_of(head) == {}
    assert json.loads(worker.cluster_file.read_text(encoding="utf-8"))["worker"] is None

    revoked = requests.post(
        f"{head.url}/v1/cluster/heartbeat",
        headers={"Authorization": f"Bearer {secret}"},
        json={"seq": 99, "epoch": "replay"},
        timeout=10,
    )
    assert revoked.status_code == 401

    # The operator force-remove path works on a freshly re-joined member.
    minted = head.cli("cluster", "token")
    token_file.write_text(TOKEN_RE.search(minted.stdout).group(0), encoding="utf-8")
    rejoined = worker.cli("join", head.url, "--token-file", str(token_file))
    assert rejoined.returncode == 0, rejoined.stdout + rejoined.stderr
    new_id = wait_until(
        lambda: member_of(head).get("id"), message="the worker to re-join"
    )

    removed = head.request("DELETE", f"/v1/cluster/members/{new_id}")
    assert removed.status_code == 200, removed.text
    assert member_of(head) == {}
