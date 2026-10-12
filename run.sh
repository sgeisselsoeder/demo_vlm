#!/usr/bin/env bash
# Startet die VLM-Live-Demo. Beispiele:
#   ./run.sh                                   # Kamera 0, Qwen3-VL 2B
#   ./run.sh --camera 4 --model qwen3-vl-4b-nf4
#   ./run.sh --host 0.0.0.0                    # im Netz erreichbar
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || uv sync
# Xet-Downloads brechen in manchen Netzen ab; klassisches HTTP ist robuster
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# direkt den venv-Python starten (das Snap-"uv run" verändert die Umgebung des Kindprozesses)
exec .venv/bin/python -u -m vlm_demo.app "$@"
