# SPDX-License-Identifier: Apache-2.0
"""Per-node capability checks for cluster serving.

Every check here answers one question: can *this* machine take part in a
distributed run, and if not, what does the operator have to do about it?

The distinction that matters is between the two interconnects:

- The TCP `ring` backend needs nothing but an IP route. It works on any Mac,
  any macOS, over wifi or ethernet.
- The `jaccl` backends need RDMA over Thunderbolt, which is far more demanding:
  macOS 26.2+, Thunderbolt 5 silicon, RDMA armed once per machine from the
  Recovery OS, and Thunderbolt Bridge switched off because `bridge0` claims the
  same interfaces the RDMA driver wants.

So a failing check is not necessarily fatal. `Preflight.best_backend()` reports
the fastest transport this node actually qualifies for, and the caller decides
whether to run degraded or refuse.

Observed on macOS 27.0 (build 26A5388g), M5 Max and M3 Ultra:

- `rdma_ctl status` prints exactly `enabled` or `disabled`.
- `rdma_ctl enable` outside Recovery prints
  "rdma_ctl: This tool needs to be executed from Recovery OS." and still
  **exits 0**, so the exit status cannot be trusted - parse the output.
- `ibv_devices` names its devices after the Thunderbolt ports: `rdma_en1`
  corresponds to the `en1` interface that `networksetup` calls "Thunderbolt 1".
  With RDMA disabled the command still succeeds and prints only its header.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Backend = Literal["jaccl", "jaccl-ring", "ring"]

# RDMA over Thunderbolt was introduced in macOS 26.2 (Apple TN3205).
MIN_RDMA_MACOS = (26, 2)

# Anything slower than this on a Thunderbolt link means we negotiated down to
# TB4 or below, which has no RDMA at all. TB5 runs 80 Gb/s symmetric; the
# advertised 120 Gb/s is the asymmetric display mode, not a data-path rate.
MIN_TB5_GBPS = 80

_COMMAND_TIMEOUT_S = 15


def _run(*argv: str) -> tuple[int, str]:
    """Run a command, returning (returncode, combined output).

    Never raises: a missing binary or a timeout is reported as a non-zero
    return code so callers can treat "tool absent" and "tool said no" the same.
    """
    if shutil.which(argv[0]) is None:
        return 127, f"{argv[0]}: not found"
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        logger.warning("preflight: %s timed out", argv[0])
        return 124, f"{argv[0]}: timed out"
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("preflight: %s failed: %s", argv[0], exc)
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@dataclass(frozen=True)
class Check:
    """One preflight question and its answer."""

    name: str
    ok: bool
    detail: str
    # How the operator fixes it. Empty when `ok`, or when nothing can be done
    # (e.g. the silicon is simply too old).
    remedy: str = ""


@dataclass
class Preflight:
    """The full capability report for this node."""

    macos: tuple[int, int]
    chip: str
    rdma_enabled: bool
    rdma_devices: list[str]
    bridged_interfaces: list[str]
    tb_max_gbps: int
    checks: list[Check] = field(default_factory=list)

    @property
    def rdma_ready(self) -> bool:
        """True when this node can actually carry JACCL traffic."""
        return (
            self.macos >= MIN_RDMA_MACOS
            and self.rdma_enabled
            and bool(self.rdma_devices)
            and not self.bridged_interfaces
            and self.tb_max_gbps >= MIN_TB5_GBPS
        )

    def best_backend(self, *, mesh_complete: bool = True) -> Backend:
        """The fastest transport this node qualifies for.

        `mesh_complete` comes from the cluster-wide topology analysis - a node
        can be perfectly RDMA-capable and still be stuck on `jaccl-ring`
        because somebody did not plug in enough cables.
        """
        if not self.rdma_ready:
            return "ring"
        return "jaccl" if mesh_complete else "jaccl-ring"

    def blockers(self) -> list[Check]:
        """Failed checks, in the order they should be shown to an operator."""
        return [c for c in self.checks if not c.ok]


def _macos_version() -> tuple[int, int]:
    parts = platform.mac_ver()[0].split(".")
    if not parts or not parts[0]:
        return (0, 0)
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)


def _chip() -> str:
    rc, out = _run("sysctl", "-n", "machdep.cpu.brand_string")
    return out.strip() if rc == 0 else "unknown"


def rdma_status() -> bool:
    """Whether RDMA over Thunderbolt is armed on this machine.

    Arming is a one-time, per-machine action performed from the Recovery OS.
    It survives reboots.
    """
    rc, out = _run("rdma_ctl", "status")
    if rc != 0:
        return False
    return out.strip().splitlines()[-1].strip().lower() == "enabled" if out.strip() else False


def ibv_devices() -> list[str]:
    """RDMA device names, e.g. ``["rdma_en1", "rdma_en2", "rdma_en7"]``.

    Returns empty when RDMA is disarmed - the command still succeeds and emits
    only its two header lines, so we key off the `rdma_` prefix rather than
    trying to skip a fixed number of lines.
    """
    rc, out = _run("ibv_devices")
    if rc != 0:
        return []
    devices = []
    for line in out.splitlines():
        token = line.strip().split()
        if token and token[0].startswith("rdma_"):
            devices.append(token[0])
    return sorted(devices)


def thunderbolt_bridge_members() -> list[str]:
    """Thunderbolt interfaces currently captured by macOS's `bridge0`.

    `bridge0` and the RDMA driver both want the `enN` Thunderbolt interfaces,
    and the bridge wins. What matters is not whether `bridge0` exists but
    whether it holds any ports, so we return the member list.

    Enabling Internet Sharing over Thunderbolt creates this bridge. Turning
    Internet Sharing back off drops the bridge's IP addresses but leaves the
    interfaces enslaved (observed on macOS 27.0), so the conflict outlives the
    setting that caused it. It also returns by itself after a reboot, which is
    why this is re-checked on every cluster formation rather than once at setup.
    """
    rc, out = _run("ifconfig", "bridge0")
    if rc != 0:
        return []
    return re.findall(r"member:\s*(en\d+)", out)


def thunderbolt_max_gbps() -> int:
    """Fastest Thunderbolt port speed this machine advertises, in Gb/s.

    Reads `system_profiler SPThunderboltDataType`, which reports both connected
    links ("Speed: 80 Gb/s") and idle ports ("Speed: Up to 120 Gb/s"). We take
    the maximum across all of them: an idle TB5 port still proves the machine
    has TB5 silicon, which is what the RDMA requirement is really about.
    """
    rc, out = _run("system_profiler", "SPThunderboltDataType")
    if rc != 0:
        return 0
    speeds = [int(m) for m in re.findall(r"Speed:\s*(?:Up to\s*)?(\d+)\s*Gb/s", out)]
    return max(speeds, default=0)


def run() -> Preflight:
    """Probe this node and return its capability report."""
    macos = _macos_version()
    chip = _chip()
    enabled = rdma_status()
    devices = ibv_devices()
    bridged = thunderbolt_bridge_members()
    gbps = thunderbolt_max_gbps()

    checks = [
        Check(
            name="macos_version",
            ok=macos >= MIN_RDMA_MACOS,
            detail=f"macOS {macos[0]}.{macos[1]}",
            remedy=(
                ""
                if macos >= MIN_RDMA_MACOS
                else f"RDMA over Thunderbolt needs macOS "
                f"{MIN_RDMA_MACOS[0]}.{MIN_RDMA_MACOS[1]} or later. "
                f"The TCP ring backend still works on this version."
            ),
        ),
        Check(
            name="thunderbolt5",
            ok=gbps >= MIN_TB5_GBPS,
            detail=f"{chip}, Thunderbolt up to {gbps} Gb/s",
            remedy=(
                ""
                if gbps >= MIN_TB5_GBPS
                else "This Mac has no Thunderbolt 5 port. Thunderbolt 4 and "
                "earlier carry no RDMA; use the TCP ring backend."
            ),
        ),
        Check(
            name="rdma_armed",
            ok=enabled,
            detail="rdma_ctl reports " + ("enabled" if enabled else "disabled"),
            remedy=(
                ""
                if enabled
                else "Boot this Mac into the Recovery OS, run `rdma_ctl enable` "
                "in Terminal, then reboot. This is a one-time step per machine "
                "and cannot be done from the running system."
            ),
        ),
        Check(
            name="rdma_devices",
            ok=bool(devices),
            detail=(
                ", ".join(devices) if devices else "ibv_devices lists no devices"
            ),
            remedy=(
                ""
                if devices
                else "No RDMA devices enumerated. Arm RDMA from the Recovery OS "
                "and make sure Thunderbolt Bridge is off."
            ),
        ),
        Check(
            name="thunderbolt_bridge_off",
            ok=not bridged,
            detail=(
                f"bridge0 holds {', '.join(bridged)}"
                if bridged
                else "no Thunderbolt interfaces are bridged"
            ),
            remedy=(
                "Thunderbolt Bridge is holding the interfaces RDMA needs. Turn "
                "it off in System Settings > Network, and turn off Internet "
                "Sharing over Thunderbolt if it is on - that is what creates "
                "the bridge. Disabling Internet Sharing alone leaves the "
                "interfaces enslaved until the bridge is removed."
                if bridged
                else ""
            ),
        ),
    ]

    return Preflight(
        macos=macos,
        chip=chip,
        rdma_enabled=enabled,
        rdma_devices=devices,
        bridged_interfaces=bridged,
        tb_max_gbps=gbps,
        checks=checks,
    )
