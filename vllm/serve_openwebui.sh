#!/bin/bash
#===============================================================================
# serve_openwebui.sh — launch an Open WebUI chat front-end as a small CPU-only
# LSF job pointed at a running vLLM server, and PRINT THE ACCESS URL.
#
# Open WebUI is a stateful web app (no GPU). It runs on a CPU node via rootless
# Podman and talks to your model's OpenAI endpoint (e.g. a serve_vllm.sh job).
# Because LSF picks the node at dispatch time, this wrapper waits for the job to
# start, then reports the URL + the SSH tunnel to reach it.
#
# Usage:
#   ./serve_openwebui.sh --model-url http://i01u02:8000/v1
#   ./serve_openwebui.sh --model-url http://i01u22:8000/v1 --port 8080 --queue local
#
# Options (defaults in []):
#   --model-url URL     vLLM OpenAI base URL to point at (required, e.g. http://<node>:8000/v1)
#   --port N            Web UI port                         [8080]
#   --queue Q           CPU LSF queue                       [local]
#   --walltime H:MM     LSF walltime                        [24:00]
#   --cpus N            CPU slots (1 slot ~= 16 G RAM)      [1]
#   --data-dir PATH     Open WebUI data dir                 [/scratch/$USER/open-webui]
#   --login-host HOST   host to tunnel through in the hint  [login.int.janelia.org]
#   --image REF         Open WebUI image  [ghcr.io/open-webui/open-webui:main]
#===============================================================================
set -euo pipefail

MODEL_URL=""; PORT=8080; QUEUE="local"; WALLTIME="24:00"; CPUS=1   # 1 slot ~= 16 G RAM, ample for the UI
DATA_DIR="/scratch/\$USER/open-webui-vllm"   # dedicated dir; avoids inheriting a pre-existing
                                             # personal ~/open-webui DB (stale models/personas).
                                             # \$USER expands on the compute node
LOGIN_HOST="login.int.janelia.org"
DOMAIN=".int.janelia.org"                    # appended to the node short-name for routable URLs
IMAGE="ghcr.io/open-webui/open-webui:main"
JOB_NAME="openwebui"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-url=*) MODEL_URL="${1#*=}"; shift ;;   --model-url) MODEL_URL="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;             --port) PORT="$2"; shift 2 ;;
    --queue=*) QUEUE="${1#*=}"; shift ;;           --queue) QUEUE="$2"; shift 2 ;;
    --walltime=*) WALLTIME="${1#*=}"; shift ;;     --walltime) WALLTIME="$2"; shift 2 ;;
    --cpus=*) CPUS="${1#*=}"; shift ;;             --cpus) CPUS="$2"; shift 2 ;;
    --data-dir=*) DATA_DIR="${1#*=}"; shift ;;     --data-dir) DATA_DIR="$2"; shift 2 ;;
    --login-host=*) LOGIN_HOST="${1#*=}"; shift ;; --login-host) LOGIN_HOST="$2"; shift 2 ;;
    --domain=*) DOMAIN="${1#*=}"; shift ;;         --domain) DOMAIN="$2"; shift 2 ;;
    --image=*) IMAGE="${1#*=}"; shift ;;           --image) IMAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[ -n "$MODEL_URL" ] || { echo "ERROR: --model-url is required (e.g. http://i01u02:8000/v1)" >&2; exit 1; }

OUTPUT_DIR="$(cd "$(dirname "$0")" && { git rev-parse --show-toplevel 2>/dev/null || pwd; })/output"
mkdir -p "$OUTPUT_DIR"

echo "Submitting Open WebUI -> ${MODEL_URL}  (queue=$QUEUE, port=$PORT)"
SUBMIT=$(bsub -q "$QUEUE" -J "$JOB_NAME" -n "$CPUS" -W "$WALLTIME" \
    -o "$OUTPUT_DIR/${JOB_NAME}_%J.out" -e "$OUTPUT_DIR/${JOB_NAME}_%J.err" <<EOF
unset XDG_RUNTIME_DIR
mkdir -p /scratch/\$USER/podman-run /scratch/\$USER/podman-storage ${DATA_DIR}
podman system migrate 2>/dev/null || true
podman info >/dev/null 2>&1 || podman system reset -f
trap 'podman rm -f ${JOB_NAME} 2>/dev/null; pkill -u \$USER catatonit 2>/dev/null' EXIT
echo "OPEN-WEBUI on \$(hostname -s):${PORT} -> ${MODEL_URL}"
podman run --rm --name ${JOB_NAME} --network=host \
  -e WEBUI_AUTH=False -e ENABLE_OLLAMA_API=False \
  -e OPENAI_API_BASE_URL=${MODEL_URL} -e OPENAI_API_KEY=dummy \
  -e PORT=${PORT} \
  -v ${DATA_DIR}:/app/backend/data \
  ${IMAGE}
EOF
)
echo "$SUBMIT"
JOBID=$(echo "$SUBMIT" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1)
[ -n "$JOBID" ] || { echo "ERROR: submission failed"; exit 1; }

echo -n "Waiting for job $JOBID to start"
NODE=""
for i in $(seq 1 200); do
    st=$(bjobs -o "stat" "$JOBID" 2>/dev/null | tail -1 | tr -d ' ')
    if [ "$st" = "RUN" ]; then
        NODE=$(bjobs -o "exec_host" "$JOBID" 2>/dev/null | tail -1 | tr -d ' ' | sed 's/^[0-9]*\*//; s/:.*//')
        break
    fi
    [ "$st" = "EXIT" ] || [ "$st" = "DONE" ] && { echo; echo "ERROR: job $st before starting"; exit 1; }
    echo -n "."; sleep 3
done
echo
[ -n "$NODE" ] || { echo "Job not running yet; check: bjobs $JOBID"; exit 0; }

cat <<MSG
================================================================
Open WebUI job $JOBID is running on: $NODE
  Backend model API : ${MODEL_URL}
  (the UI may take ~30-60 s more to finish first-time DB init)

Access it via an SSH tunnel:
  ssh -L ${PORT}:${NODE}${DOMAIN}:${PORT} ${LOGIN_HOST}
  then open:  http://localhost:${PORT}

Direct URL (only if your machine can route to the compute node):
  http://${NODE}${DOMAIN}:${PORT}

Stop it with:  bkill $JOBID
================================================================
MSG
