# SPDX-License-Identifier: Apache-2.0
"""Peer discovery for oMLX cluster mode.

Before several oMLX daemons can shard one large model between them, they have
to find each other. There is no central registry and no SSH: every Mac in the
workgroup already runs the daemon, so discovery is peer-to-peer over Bonjour
(service type ``_omlx._tcp`` on the local domain) - the same protocol AirPlay
and printer sharing use, so it needs no firewall rule this LAN does not
already carry.

Backend: dns-sd, not zeroconf
------------------------------
`zeroconf` is not a dependency of this project - checked `pyproject.toml` (no
match) and the venv (`uv run python -c "import zeroconf"` raises
`ModuleNotFoundError`) - and adding one just for discovery was out of scope.
macOS ships the `dns-sd` command on every machine this targets, so
`DnsSdBackend` shells out to it instead:

- `dns-sd -R <name> _omlx._tcp local <port> <k=v> ...` to advertise. This
  process is the registration; it never exits on its own, so it is kept
  running for as long as we are advertising and torn down in `stop()`.
- `dns-sd -B _omlx._tcp local` to browse for peers. Rather than keep one
  long-lived process alive and stream its output forever (which needs a
  reader thread, and does not repeat an `Add` event for a peer that is still
  present and unchanged), discovery re-runs a short, time-boxed `-B` scan
  every `poll_interval`. A fresh scan re-triggers an mDNS query and gets a
  fresh answer from every responder still alive - exactly the "did anyone go
  quiet" signal `last_seen` needs, without having to parse dns-sd's
  more-coming flag bits.
- `dns-sd -L <instance> _omlx._tcp local` to resolve a newly-seen instance's
  host, port and TXT record. Resolved once per instance, not on every scan -
  the fields we advertise (version, chip, RAM) do not change mid-session.

All three command shapes and their exact output formats below were confirmed
by running real `dns-sd -R` / `-B` / `-L` against each other on this machine,
not assumed from documentation. Notably: `-L` backslash-escapes literal
spaces inside a TXT value (``chip=Apple\\ M3\\ Ultra``), and its instance-name
column can itself contain spaces, so both parsers below are written to expect
that rather than assume whitespace-delimited fields throughout.

`DiscoveryBackend` is the seam: everything above lives behind three methods
(`advertise` / `stop_advertising` / `scan`), so a `zeroconf`-based backend can
be dropped in later - e.g. for non-macOS support - without touching
`ClusterDiscovery`.

Security
--------
Peers only belong to the same cluster if they were started with the same
`cluster_key` (an operator-supplied shared secret, not defined by this
module). The TXT record is broadcast in cleartext to the whole LAN, so it
never carries the key itself - only `fingerprint()`, a truncated SHA-256 hex
digest. `matches_fingerprint()` lets a caller reject a peer whose fingerprint
does not match before it is ever treated as part of the cluster.

Safe when cluster mode is disabled
-----------------------------------
Constructing a `ClusterDiscovery` opens no socket, starts no thread, and
spawns no subprocess. All of that begins in `start()` and stops - thread
joined, subprocesses killed - in `stop()`, which is idempotent and bounded by
a timeout so a wedged `dns-sd` process can never block shutdown.

Failure modes this module handles rather than assumes away
------------------------------------------------------------
- `dns-sd` missing or the subprocess failing to spawn: `available()` reports
  it up front, `start()` raises `DiscoveryUnavailableError` rather than silently
  running with no peers ever found.
- No network / no responders: a scan simply returns nothing; there are no
  peers, not an error.
- Two machines advertising the same `node_id` (e.g. a cloned image or a
  copy-pasted hardware UUID): logged as a warning, keeping the most recently
  seen host - proportionate for a LAN of a handful of desks, not a
  distributed consensus problem.
- A peer that vanishes without deregistering (crash, unplugged cable, closed
  lid): it stops answering scans, `last_seen` stops advancing, and it is
  expired and reported as `"left"` once `expire_after` has passed. No goodbye
  packet is required.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_omlx._tcp"
DOMAIN = "local"

# How often a live browse scan re-runs, and how long each dns-sd invocation is
# allowed to answer before we read what it printed and kill it.
DEFAULT_POLL_INTERVAL_S = 15.0
_BROWSE_WINDOW_S = 3.0
_RESOLVE_WINDOW_S = 3.0

# A peer not re-confirmed by a scan for this long is considered gone. This is
# what stands in for a goodbye packet from a peer that vanished uncleanly.
DEFAULT_EXPIRE_AFTER_S = 45.0

_SUBPROCESS_JOIN_TIMEOUT_S = 5.0
_THREAD_JOIN_TIMEOUT_S = _BROWSE_WINDOW_S + _RESOLVE_WINDOW_S + _SUBPROCESS_JOIN_TIMEOUT_S
_PROBE_TIMEOUT_S = 5.0

# `_now` exists so tests can monkeypatch a deterministic clock instead of
# racing real wall-clock time to exercise the expiry path.
_now: Callable[[], float] = time.monotonic


class DiscoveryUnavailableError(RuntimeError):
    """No usable Bonjour backend on this machine."""


# =============================================================================
# Cluster-key fingerprinting
# =============================================================================


def fingerprint(cluster_key: str, length: int = 16) -> str:
    """Non-reversible fingerprint of a cluster key, safe to broadcast.

    Truncated to `length` hex characters (64 bits by default) - enough to
    rule out accidental cross-talk between two unrelated clusters on the same
    LAN, not a cryptographic commitment. The full key never leaves this
    process; only this digest goes in the TXT record.
    """
    return hashlib.sha256(cluster_key.encode("utf-8")).hexdigest()[:length]


def matches_fingerprint(cluster_key: str, peer_fingerprint: str) -> bool:
    """Whether a discovered peer's advertised fingerprint matches ours."""
    return fingerprint(cluster_key) == peer_fingerprint


# =============================================================================
# What a peer advertises
# =============================================================================


@dataclass(frozen=True)
class PeerInfo:
    """Everything one peer advertises about itself in its TXT record."""

    node_id: str
    version: str
    port: int
    chip: str
    ram_gb: int
    key_fingerprint: str


def encode_txt(info: PeerInfo) -> dict[str, str]:
    """`PeerInfo` -> the key/value pairs handed to `dns-sd -R`."""
    return {
        "node_id": info.node_id,
        "version": info.version,
        "port": str(info.port),
        "chip": info.chip,
        "ram_gb": str(info.ram_gb),
        "key_fingerprint": info.key_fingerprint,
    }


def decode_txt(txt: dict[str, str]) -> PeerInfo | None:
    """The inverse of `encode_txt`.

    Returns `None` on a malformed record rather than raising - a peer running
    a future or older version of this module should degrade to "not
    discovered", not crash discovery for every other peer on the LAN.
    """
    try:
        return PeerInfo(
            node_id=txt["node_id"],
            version=txt.get("version", ""),
            port=int(txt["port"]),
            chip=txt.get("chip", "unknown"),
            ram_gb=int(txt.get("ram_gb", "0")),
            key_fingerprint=txt.get("key_fingerprint", ""),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("discovery: malformed TXT record %r: %s", txt, exc)
        return None


# =============================================================================
# The peer table
# =============================================================================

PeerEventType = Literal["joined", "left"]


@dataclass
class Peer:
    """A discovered peer and when a scan last confirmed it is still there."""

    info: PeerInfo
    host: str
    last_seen: float  # `_now()`, i.e. a `time.monotonic()` timestamp


ObserverCallback = Callable[[PeerEventType, Peer], None]


# =============================================================================
# Backend seam
# =============================================================================


class DiscoveryBackend(Protocol):
    """The seam between peer-table bookkeeping and the underlying transport.

    A backend is a dumb transport: it does not know about `cluster_key`,
    expiry, or join/leave semantics - that all lives in `ClusterDiscovery`.
    """

    def available(self) -> bool:
        """Whether this backend can actually be used on this machine."""
        ...

    def advertise(self, node_id: str, port: int, txt: dict[str, str]) -> None:
        """Start advertising this node. No-op if already advertising."""
        ...

    def stop_advertising(self) -> None:
        """Idempotent; safe even if `advertise()` was never called."""
        ...

    def scan(self) -> dict[str, tuple[str, dict[str, str]]]:
        """One browse-and-resolve pass over the LAN.

        Returns ``{instance_name: (host, txt)}`` for every peer answering
        right now. Blocking, bounded by this module's browse/resolve windows.
        """
        ...

    def stop(self) -> None:
        """Release anything `scan()` might have left running. Idempotent."""
        ...


def _terminate(proc: subprocess.Popen, timeout: float = _SUBPROCESS_JOIN_TIMEOUT_S) -> None:
    """Stop a subprocess, escalating to SIGKILL, without ever hanging."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("discovery: process %s would not die", proc.pid)


def _run_windowed(argv: list[str], window_s: float) -> str:
    """Run a streaming `dns-sd` command for `window_s` seconds, return its output.

    `-B` and `-L` never exit on their own, so this is the pattern for both:
    start it, let it answer for a fixed window, then stop it and read what it
    wrote. `communicate()` (rather than a separate `wait()`) drains stdout as
    it stops the process, so output cannot deadlock a full pipe.
    """
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except OSError as exc:
        logger.warning("discovery: failed to start %s: %s", argv[0], exc)
        return ""
    time.sleep(window_s)
    if proc.poll() is None:
        proc.terminate()
    try:
        output, _ = proc.communicate(timeout=_SUBPROCESS_JOIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            output, _ = proc.communicate(timeout=_SUBPROCESS_JOIN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("discovery: %s would not die", argv[0])
            output = ""
    return output or ""


# `dns-sd` pads the hour with a *space*, not a zero, so before 10:00 the
# timestamp column is `9:04:13.991` and not `09:04:13.991`. Requiring two
# digits made peer discovery work only after 10am - it found nothing at all
# for the first ten hours of every day, and the failure looked exactly like a
# LAN with no peers on it.
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}\.\d+$")


def _parse_browse_line(line: str) -> tuple[str, str] | None:
    """One line of `dns-sd -B` output -> `(event, instance_name)`, or `None`.

    `event` is `"Add"` or `"Rmv"`. Header banners (`"DATE: ..."`,
    `"...STARTING..."`, the column header row) do not match and are skipped.
    The instance name is whatever remains after the first six columns, since
    an advertised name may itself contain spaces.
    """
    parts = line.strip().split(None, 6)
    if len(parts) < 7:
        return None
    timestamp, event, _flags, _iface, domain, service_type, instance = parts
    if not _TIME_RE.match(timestamp) or event not in ("Add", "Rmv"):
        return None
    if domain.rstrip(".") != DOMAIN or service_type.rstrip(".") != SERVICE_TYPE:
        return None
    return event, instance


_REACH_RE = re.compile(r"can be reached at\s+(?P<host>\S+?):(?P<port>\d+)\s+\(interface")


def _parse_txt_tokens(txt_line: str) -> dict[str, str]:
    """Split a `dns-sd -L` TXT line into key/value pairs.

    A value containing a literal space comes back with the space
    backslash-escaped (``chip=Apple\\ M3\\ Ultra``) - confirmed against a real
    `dns-sd -L` round trip, not assumed from documentation. Split on
    whitespace that is *not* preceded by a backslash, then undo the escape.
    """
    tokens = re.split(r"(?<!\\)\s+", txt_line.strip())
    result: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        result[key] = value.replace("\\ ", " ")
    return result


def _parse_resolve_output(output: str) -> tuple[str, dict[str, str]] | None:
    """`dns-sd -L` output -> `(host, txt)`, or `None` if nothing resolved.

    `-L` reprints the same answer once per network interface; the first
    complete (host, TXT) pair found is used and the rest are duplicates.
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        match = _REACH_RE.search(line)
        if not match or i + 1 >= len(lines):
            continue
        txt_line = lines[i + 1].strip()
        if not txt_line:
            continue
        return match.group("host").rstrip("."), _parse_txt_tokens(txt_line)
    return None


class DnsSdBackend:
    """`dns-sd`-backed implementation of `DiscoveryBackend`. See module docstring
    for why this was chosen over `zeroconf`."""

    def __init__(self) -> None:
        self._advertise_proc: subprocess.Popen | None = None

    def available(self) -> bool:
        return shutil.which("dns-sd") is not None

    def advertise(self, node_id: str, port: int, txt: dict[str, str]) -> None:
        if self._advertise_proc is not None:
            return
        argv = ["dns-sd", "-R", node_id, SERVICE_TYPE, DOMAIN, str(port)]
        argv.extend(f"{k}={v}" for k, v in txt.items())
        try:
            self._advertise_proc = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
            )
        except OSError as exc:
            logger.warning("discovery: failed to start dns-sd -R: %s", exc)
            self._advertise_proc = None

    def stop_advertising(self) -> None:
        proc, self._advertise_proc = self._advertise_proc, None
        if proc is not None:
            _terminate(proc)

    def scan(self) -> dict[str, tuple[str, dict[str, str]]]:
        instances = self._browse_once()
        resolved: dict[str, tuple[str, dict[str, str]]] = {}
        for instance in instances:
            info = self._resolve_once(instance)
            if info is not None:
                resolved[instance] = info
        return resolved

    def _browse_once(self) -> set[str]:
        output = _run_windowed(["dns-sd", "-B", SERVICE_TYPE, DOMAIN], _BROWSE_WINDOW_S)
        instances: set[str] = set()
        for line in output.splitlines():
            parsed = _parse_browse_line(line)
            if parsed is None:
                continue
            event, instance = parsed
            if event == "Add":
                instances.add(instance)
            else:
                instances.discard(instance)
        return instances

    def _resolve_once(self, instance: str) -> tuple[str, dict[str, str]] | None:
        argv = ["dns-sd", "-L", instance, SERVICE_TYPE, DOMAIN]
        return _parse_resolve_output(_run_windowed(argv, _RESOLVE_WINDOW_S))

    def stop(self) -> None:
        self.stop_advertising()


# =============================================================================
# Best-effort local facts, for building this node's own TXT record
# =============================================================================


def default_node_id() -> str:
    """A stable id for this machine, derived from its hardware UUID.

    Falls back to the hostname if `ioreg` is unavailable - the hostname is
    not guaranteed unique across a LAN, but it beats refusing to advertise.
    """
    if shutil.which("ioreg") is not None:
        try:
            proc = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_S,
            )
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', proc.stdout)
            if match:
                return match.group(1)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("discovery: ioreg failed, falling back to hostname: %s", exc)
    return socket.gethostname()


def default_chip() -> str:
    """Best-effort chip name, e.g. `"Apple M5 Max"`."""
    if shutil.which("sysctl") is None:
        return "unknown"
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def default_ram_gb() -> int:
    """Total physical RAM in GB, rounded down. `0` if it cannot be read."""
    if shutil.which("sysctl") is None:
        return 0
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    try:
        return int(proc.stdout.strip()) // (1024**3)
    except ValueError:
        return 0


# =============================================================================
# ClusterDiscovery: advertise this node, maintain the live peer table
# =============================================================================


class ClusterDiscovery:
    """Advertises this node and maintains a live table of cluster peers.

    Safe to construct at any time, including when `cluster.enabled = false` -
    nothing opens a socket, starts a thread, or spawns a subprocess before
    `start()` is called.
    """

    def __init__(
        self,
        *,
        node_id: str,
        port: int,
        version: str,
        cluster_key: str,
        # Default to probing this machine. A peer that advertises
        # chip="unknown"/ram_gb=0 is useless for capacity planning, and the
        # helpers to detect both are right here.
        chip: str | None = None,
        ram_gb: int | None = None,
        backend: DiscoveryBackend | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        expire_after: float = DEFAULT_EXPIRE_AFTER_S,
        on_event: ObserverCallback | None = None,
    ) -> None:
        self._node_id = node_id
        self._port = port
        self._version = version
        self._cluster_key = cluster_key
        # Left unresolved until the TXT record is built. Probing the machine
        # means running sysctl, and construction must stay free of
        # subprocesses so it is safe when cluster mode is disabled.
        self._chip = chip
        self._ram_gb = ram_gb
        self._backend = backend if backend is not None else DnsSdBackend()
        self._poll_interval = poll_interval
        self._expire_after = expire_after
        self._on_event = on_event

        self._peers: dict[str, Peer] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def peers(self) -> list[Peer]:
        """A snapshot of currently-known peers."""
        with self._lock:
            return list(self._peers.values())

    def start(self) -> None:
        """Begin advertising this node and browsing for peers.

        Raises `DiscoveryUnavailableError` if no backend can run on this machine -
        the caller decides whether that should disable cluster mode entirely
        or just proceed single-node.
        """
        if self._started:
            return
        if not self._backend.available():
            raise DiscoveryUnavailableError(
                "no Bonjour backend available (dns-sd not found on PATH)"
            )
        self._backend.advertise(self._node_id, self._port, self._own_txt())
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="omlx-cluster-discovery", daemon=True
        )
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        """Stop advertising and browsing. Idempotent and non-hanging."""
        if not self._started:
            return
        self._started = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "discovery: browse thread did not stop within %.1fs",
                    _THREAD_JOIN_TIMEOUT_S,
                )
            self._thread = None
        self._backend.stop_advertising()
        self._backend.stop()

    def _own_txt(self) -> dict[str, str]:
        # Resolve the hardware description on first use rather than at
        # construction, and cache it - peers need a real chip and RAM figure
        # to plan a shard, and "unknown"/0 would be worse than useless.
        if self._chip is None:
            self._chip = default_chip()
        if self._ram_gb is None:
            self._ram_gb = default_ram_gb()
        return encode_txt(
            PeerInfo(
                node_id=self._node_id,
                version=self._version,
                port=self._port,
                chip=self._chip,
                ram_gb=self._ram_gb,
                key_fingerprint=fingerprint(self._cluster_key),
            )
        )

    def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                self._scan_and_reconcile()
            except Exception:  # noqa: BLE001 - one bad scan must not kill discovery
                logger.exception("discovery: scan failed")
            self._stop_event.wait(self._poll_interval)

    def _scan_and_reconcile(self) -> None:
        seen = self._backend.scan()
        now = _now()
        with self._lock:
            for host, txt in seen.values():
                self._reconcile_one(host, txt, now)
            self._expire(now)

    def _reconcile_one(self, host: str, txt: dict[str, str], now: float) -> None:
        info = decode_txt(txt)
        if info is None or info.node_id == self._node_id:
            return
        existing = self._peers.get(info.node_id)
        if existing is None:
            peer = Peer(info=info, host=host, last_seen=now)
            self._peers[info.node_id] = peer
            self._notify("joined", peer)
            return
        if existing.host != host:
            logger.warning(
                "discovery: node_id %r advertised by both %s and %s - "
                "keeping the most recently seen host; check for a cloned "
                "machine or a duplicated hardware UUID",
                info.node_id,
                existing.host,
                host,
            )
        existing.host = host
        existing.info = info
        existing.last_seen = now

    def _expire(self, now: float) -> None:
        stale = [
            node_id
            for node_id, peer in self._peers.items()
            if now - peer.last_seen > self._expire_after
        ]
        for node_id in stale:
            peer = self._peers.pop(node_id)
            self._notify("left", peer)

    def _notify(self, event: PeerEventType, peer: Peer) -> None:
        logger.info(
            "discovery: peer %s node_id=%s host=%s", event, peer.info.node_id, peer.host
        )
        if self._on_event is None:
            return
        try:
            self._on_event(event, peer)
        except Exception:  # noqa: BLE001 - a bad observer must not break discovery
            logger.exception("discovery: observer callback raised")
