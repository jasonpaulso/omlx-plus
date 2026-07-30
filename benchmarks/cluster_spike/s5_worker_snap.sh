#!/bin/bash
# Snapshot the worker's copy of a model dir: path, size, mtime_ns, sha256 (optional).
# Usage: s5_worker_snap.sh <rel_model_dir_under_~/Models> [--sha]
# Prints one JSON object per file, sorted by path. Empty output = dir absent.
DIR="$1"
SHA="$2"
ssh -o ConnectTimeout=8 Jasons-Mac-Studio.local MDIR="\$HOME/Models/$DIR" SHA="$SHA" 'bash -s' <<'EOF'
[ -d "$MDIR" ] || exit 0
cd "$MDIR"
find . -type f | sort | while read -r f; do
  p="${f#./}"
  size=$(stat -f %z "$f")
  mtime=$(stat -f %Fm "$f" | tr -d '.')
  if [ "$SHA" = "--sha" ]; then
    h=$(shasum -a 256 "$f" | cut -d' ' -f1)
    echo "{\"path\": \"$p\", \"size\": $size, \"mtime_ns\": $mtime, \"sha256\": \"$h\"}"
  else
    echo "{\"path\": \"$p\", \"size\": $size, \"mtime_ns\": $mtime}"
  fi
done
EOF
