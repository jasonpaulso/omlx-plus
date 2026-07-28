#!/bin/bash
# S0 ring-backend bring-up + measurement run.
# Rank 0 = this machine (local M4 Max), Rank 1 = Jasons-Mac-Studio.local.
# Static TB IPs: 10.0.2.1 (local) / 10.0.2.2 (remote), per bringup.md.
#
# CRITICAL ordering: start rank 0 (listener) first, THEN the peer. A peer
# starting early burns its connect window (salvage pitfall #3).
# Never TCP-probe the ring port to "check" it -- the probe is accepted as a
# peer and poisons the handshake (salvage pitfall: use process table instead).
set -euo pipefail

LOCAL_REPO=/Users/jasonschulz/Developer/Runners/omlx
REMOTE_HOST=Jasons-Mac-Studio.local
REMOTE_REPO=/Users/jasonschulz/Developer/Repos/omlx
LOCAL_IP=10.0.2.1
REMOTE_IP=10.0.2.2
RING_PORT=41100
HOSTFILE=/tmp/omlx_spike_ring_hostfile.json

echo "[sweep] killing any orphaned spike ranks on both machines"
pkill -f "benchmarks/cluster_spike/collective_spike.py" 2>/dev/null || true
ssh "$REMOTE_HOST" "pkill -f collective_spike.py" 2>/dev/null || true
sleep 1

echo "[hostfile] writing ring hostfile (rank order: local=0, remote=1)"
python3 -c "
import json
addrs = [['${LOCAL_IP}:${RING_PORT}'], ['${REMOTE_IP}:${RING_PORT}']]
open('${HOSTFILE}', 'w').write(json.dumps(addrs))
"
cat "$HOSTFILE"
scp -q "$HOSTFILE" "$REMOTE_HOST:$HOSTFILE"

LOG_R0=/tmp/omlx_spike_ring_rank0.log
LOG_R1=/tmp/omlx_spike_ring_rank1.log

echo "[rank0] starting listener locally"
(
  cd "$LOCAL_REPO"
  source .venv/bin/activate
  export MLX_RANK=0
  export MLX_HOSTFILE="$HOSTFILE"
  export MLX_METAL_FAST_SYNCH=1
  export OMLX_CLUSTER_BACKEND=ring
  python benchmarks/cluster_spike/collective_spike.py
) > "$LOG_R0" 2>&1 &
RANK0_PID=$!

sleep 3  # let rank0's listener come up before the peer tries to connect

echo "[rank1] starting peer on remote"
ssh "$REMOTE_HOST" "
  cd $REMOTE_REPO
  source .venv/bin/activate
  export MLX_RANK=1
  export MLX_HOSTFILE=$HOSTFILE
  export MLX_METAL_FAST_SYNCH=1
  export OMLX_CLUSTER_BACKEND=ring
  python benchmarks/cluster_spike/collective_spike.py
" > "$LOG_R1" 2>&1 &
RANK1_PID=$!

wait $RANK0_PID
R0_STATUS=$?
wait $RANK1_PID
R1_STATUS=$?

echo "=== rank0 exit=$R0_STATUS log ==="
cat "$LOG_R0"
echo "=== rank1 exit=$R1_STATUS log (tail) ==="
tail -20 "$LOG_R1"

if [ $R0_STATUS -ne 0 ] || [ $R1_STATUS -ne 0 ]; then
  echo "[FAIL] ring run failed"
  exit 1
fi
echo "[OK] ring run complete"
