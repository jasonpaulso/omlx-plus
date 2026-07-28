#!/bin/bash
# S0 Task E: mlx_lm.share probe -- real model transfer node-to-node + interruption test.
# Uses mlx_lm.share's own self-launcher (launch_ring), driven by a
# "cluster description" hostfile (distinct format from the raw MLX_HOSTFILE
# array -- see hostfile.py module docstring on the two hostfile meanings).
set -uo pipefail

LOCAL_REPO=/Users/jasonschulz/Developer/Runners/omlx
REMOTE_HOST=Jasons-Mac-Studio.local
LOCAL_SELF=127.0.0.1  # mlx's RemoteProcess special-cases exactly this string to skip ssh
LOCAL_IP=10.0.2.1
REMOTE_IP=10.0.2.2
MODEL=mlx-community/Qwen3.6-27B-OptiQ-4bit   # ~22GB, present on local only (genuine presence gap)
HOSTFILE=/tmp/omlx_spike_share_hostfile.json

cd "$LOCAL_REPO"
source .venv/bin/activate

python3 -c "
import json
hf = {
  'backend': 'ring',
  'envs': [],
  'hosts': [
    {'ssh': '${LOCAL_SELF}', 'ips': ['${LOCAL_IP}'], 'rdma': []},
    {'ssh': '${REMOTE_HOST}', 'ips': ['${REMOTE_IP}'], 'rdma': []},
  ],
}
open('${HOSTFILE}', 'w').write(json.dumps(hf))
"
cat "$HOSTFILE"; echo

echo "=== full transfer ==="
time python -m mlx_lm share --model "$MODEL" --hostfile "$HOSTFILE"
