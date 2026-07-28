#!/bin/bash
# S0 jaccl-backend bring-up + measurement run.
# Rank 0 = this machine (local M4 Max, TB iface en2), Rank 1 = Jasons-Mac-Studio.local
# (TB iface en4, enslaved to bridge0).
#
# CRITICAL: JACCL leaks a kernel protection domain per init()/teardown cycle;
# recovery is reboot-only. This script must be run start-to-finish as ONE
# clean attempt. Budget: 2 clean attempts total for jaccl in S0 (spec stop
# condition). Do not loop/retry casually.
set -euo pipefail

LOCAL_REPO=/Users/jasonschulz/Developer/Runners/omlx
REMOTE_HOST=Jasons-Mac-Studio.local
REMOTE_REPO=/Users/jasonschulz/Developer/Repos/omlx
LOCAL_IP=10.0.2.1
COORD_PORT=41200
IBV_FILE=/tmp/omlx_spike_ibv_devices.json

echo "[sweep] killing any orphaned spike ranks on both machines"
pkill -f "benchmarks/cluster_spike/collective_spike.py" 2>/dev/null || true
ssh "$REMOTE_HOST" "pkill -f collective_spike.py" 2>/dev/null || true
sleep 1

echo "[ibv matrix] rank0(local)=en2 rank1(remote)=en4"
python3 -c "
import json
# matrix[i][j] = rdma device on node i that reaches node j; null on diagonal
matrix = [[None, 'rdma_en2'], ['rdma_en4', None]]
open('${IBV_FILE}', 'w').write(json.dumps(matrix))
"
cat "$IBV_FILE"
scp -q "$IBV_FILE" "$REMOTE_HOST:$IBV_FILE"

LOG_R0=/tmp/omlx_spike_jaccl_rank0.log
LOG_R1=/tmp/omlx_spike_jaccl_rank1.log

echo "[rank0] starting coordinator locally"
(
  cd "$LOCAL_REPO"
  source .venv/bin/activate
  export MLX_RANK=0
  export MLX_JACCL_COORDINATOR="${LOCAL_IP}:${COORD_PORT}"
  export MLX_IBV_DEVICES="$IBV_FILE"
  export MLX_METAL_FAST_SYNCH=1
  export OMLX_CLUSTER_BACKEND=jaccl
  python benchmarks/cluster_spike/collective_spike.py
) > "$LOG_R0" 2>&1 &
RANK0_PID=$!

sleep 3  # rank 0 must be listening before the peer tries to connect

echo "[rank1] starting peer on remote"
ssh "$REMOTE_HOST" "
  cd $REMOTE_REPO
  source .venv/bin/activate
  export MLX_RANK=1
  export MLX_JACCL_COORDINATOR=${LOCAL_IP}:${COORD_PORT}
  export MLX_IBV_DEVICES=$IBV_FILE
  export MLX_METAL_FAST_SYNCH=1
  export OMLX_CLUSTER_BACKEND=jaccl
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
  echo "[FAIL] jaccl run failed"
  exit 1
fi
echo "[OK] jaccl run complete"
