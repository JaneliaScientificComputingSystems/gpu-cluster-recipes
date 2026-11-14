#!/bin/bash
#===============================================================================
# serve_openwebui.sh — launch an Open WebUI chat front-end as a small CPU-only
# LSF job pointed at a running vLLM server, and PRINT THE ACCESS URL.
#
# Open WebUI is a stateful web app (no GPU). It runs on a CPU node via rootless
# Podman and talks to your model's OpenAI endpoint (e.g. a serve_vllm.sh job).
# Because LSF picks the node at dispatch time, this wrapper waits for the job to
# start, then reports the URL. Compute nodes are reachable directly by hostname
# from the internal network and VPN — no SSH tunnel — which is also why the UI
# asks for a login by default.
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
#   --data-dir PATH     Open WebUI data dir   [/scratch/$USER/open-webui-vllm-auth,
#                                              or /scratch/$USER/open-webui-vllm with --no-auth]
#   --login-host HOST   accepted but ignored (no tunnel is needed; kept for old command lines)
#   --image REF         Open WebUI image  [ghcr.io/open-webui/open-webui:main]
#   --api-key-file PATH file holding the model server's API key (the .apikey file that
#                       serve_vllm.sh prints). Read on the compute node, never put on a
#                       command line. Without it, $VLLM_API_KEY from your environment
#                       is used if set, else no key (server started with --no-api-key).
#   --no-auth           no login page: anyone who can reach the port is an admin
#                       (old behaviour). Default is login required.
#
# Login (default): the FIRST account created on a fresh data dir becomes the admin, so
# open the UI and sign up right away; later sign-ups wait for that admin's approval.
# Accounts live in --data-dir on the node's /scratch: the same node reuses them, a
# different node (or a purged /scratch) starts fresh with a new sign-up.
#===============================================================================
set -euo pipefail

MODEL_URL=""; PORT=8080; QUEUE="local"; WALLTIME="24:00"; CPUS=1   # 1 slot ~= 16 G RAM, ample for the UI
DATA_DIR=""                                  # default set below (depends on --no-auth)
API_KEY_FILE=""; AUTH=true
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
    --api-key-file=*) API_KEY_FILE="${1#*=}"; shift ;; --api-key-file) API_KEY_FILE="$2"; shift 2 ;;
    --no-auth) AUTH=false; shift ;;
    -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[ -n "$MODEL_URL" ] || { echo "ERROR: --model-url is required (e.g. http://i01u02:8000/v1)" >&2; exit 1; }

# Dedicated data dirs (\$USER expands on the compute node): not a pre-existing personal
# ~/open-webui DB (stale models/personas), and a SEPARATE dir per auth mode — a DB created
# without login holds a built-in default admin account and must not be reused with login on.
if [ -z "$DATA_DIR" ]; then
    if [ "$AUTH" = true ]; then DATA_DIR="/scratch/\$USER/open-webui-vllm-auth"
    else                        DATA_DIR="/scratch/\$USER/open-webui-vllm"; fi
fi
if [ "$AUTH" = true ]; then WEBUI_AUTH=True; else WEBUI_AUTH=False; fi
case "$API_KEY_FILE" in ""|/*) ;; *) API_KEY_FILE="$PWD/$API_KEY_FILE" ;; esac   # the job runs elsewhere
[ -z "$API_KEY_FILE" ] || [ -z "${VLLM_API_KEY:-}" ] || unset VLLM_API_KEY       # the file wins

OUTPUT_DIR="$(cd "$(dirname "$0")" && { git rev-parse --show-toplevel 2>/dev/null || pwd; })/output"
mkdir -p "$OUTPUT_DIR"

echo "Submitting Open WebUI -> ${MODEL_URL}  (queue=$QUEUE, port=$PORT, login=$AUTH)"
SUBMIT=$(bsub -q "$QUEUE" -J "$JOB_NAME" -n "$CPUS" -W "$WALLTIME" \
    -o "$OUTPUT_DIR/${JOB_NAME}_%J.out" -e "$OUTPUT_DIR/${JOB_NAME}_%J.err" <<EOF
unset XDG_RUNTIME_DIR
mkdir -p /scratch/\$USER/podman-run /scratch/\$USER/podman-storage ${DATA_DIR}
podman system migrate 2>/dev/null || true
podman info >/dev/null 2>&1 || podman system reset -f
# Model-server API key: read here on the node from the owner-only key file (or taken from
# VLLM_API_KEY in the submission environment) and handed to the container by environment
# only — it is never on a command line or in this job script.
KEY_FILE="${API_KEY_FILE}"
if [ -n "\$KEY_FILE" ]; then
    for i in \$(seq 1 60); do [ -s "\$KEY_FILE" ] && break; sleep 2; done   # the model job writes it at start
    [ -s "\$KEY_FILE" ] || { echo "ERROR: API key file missing or empty: \$KEY_FILE"; exit 1; }
    OPENAI_API_KEY=\$(cat "\$KEY_FILE")
else
    OPENAI_API_KEY=\${VLLM_API_KEY:-dummy}
fi
export OPENAI_API_KEY
# per-job container name, so a second UI job on this node cannot clash with or remove this one
CTR=${JOB_NAME}_\$LSB_JOBID
# only reap the rootless pause process (catatonit) when no other container of ours is left on the node
trap 'podman rm -f \$CTR 2>/dev/null; [ -n "\$(podman ps -q 2>/dev/null)" ] || pkill -u \$USER catatonit 2>/dev/null' EXIT
echo "OPEN-WEBUI on \$(hostname -s):${PORT} -> ${MODEL_URL}"
podman run --rm --name \$CTR --network=host \
  -e WEBUI_AUTH=${WEBUI_AUTH} -e ENABLE_OLLAMA_API=False \
  -e OPENAI_API_BASE_URL=${MODEL_URL} -e OPENAI_API_KEY \
  -e ENABLE_PERSISTENT_CONFIG=False \
  -e PORT=${PORT} \
  -v ${DATA_DIR}:/app/backend/data \
  ${IMAGE}
EOF
)
echo "$SUBMIT"
# "|| true": under set -e/pipefail a non-matching grep (or a transient bjobs error below) would abort silently
JOBID=$(echo "$SUBMIT" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1 || true)
[ -n "$JOBID" ] || { echo "ERROR: submission failed"; exit 1; }

echo -n "Waiting for job $JOBID to start"
NODE=""
for i in $(seq 1 200); do
    st=$(bjobs -o "stat" "$JOBID" 2>/dev/null | tail -1 | tr -d ' ' || true)
    if [ "$st" = "RUN" ]; then
        NODE=$(bjobs -o "exec_host" "$JOBID" 2>/dev/null | tail -1 | tr -d ' ' | sed 's/^[0-9]*\*//; s/:.*//' || true)
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

Open it directly from the internal network or VPN (no SSH tunnel needed):
  http://${NODE}${DOMAIN}:${PORT}

$( if [ "$AUTH" = true ]; then
     echo "Login is required. On a fresh data dir the page asks you to create an account:"
     echo "the FIRST account becomes the admin, so sign up now. If this node already holds"
     echo "your data dir from an earlier run, sign in with that account instead."
   else
     echo "WARNING: --no-auth — no login; anyone who can reach ${NODE}:${PORT} is an admin."
   fi )

Stop it with:  bkill $JOBID
================================================================
MSG
