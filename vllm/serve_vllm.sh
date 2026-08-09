#!/bin/bash
#===============================================================================
# serve_vllm.sh — serve ANY vLLM-supported model on the Janelia LSF cluster.
#
# One generic launcher for all models. Picks a container (Apptainer preferred,
# Podman optional) or a Python venv, requests GPUs on an LSF GPU queue, times the
# model load, waits for /health, and smoke-tests the OpenAI endpoint. Optionally
# also launches an Open WebUI chat front-end pointed at the model (--webui).
#
# Served model name defaults to the model dir basename (gpt-oss-20b, Kimi-K3, …).
#
# Examples:
#   ./serve_vllm.sh --model /misc/hf/openai/gpt-oss-20b --gpus 1
#   ./serve_vllm.sh --model /misc/hf/openai/gpt-oss-120b --queue gpu_h200 --gpus 2 --webui
#   ./serve_vllm.sh --model /misc/hf/Qwen/Qwen3-32B --engine venv --venv $HOME/vllm_env --gpus 2
#   # Kimi-K3: dedicated SIF, whole B300 node, tool calling + reasoning:
#   ./serve_vllm.sh --model /misc/hf/moonshotai/Kimi-K3 \
#       --sif /misc/local/singularity/vllm-kimi-k3.sif --full-node \
#       --enable-auto-tool-choice --tool-parser kimi_k3 --reasoning-parser kimi_k3 --webui
#
# Options (defaults in []):
#   --model PATH          HF weights dir (required)
#   --served-name NAME    OpenAI model id [basename of --model]
#   --engine E            apptainer | podman | venv           [apptainer]
#   --sif PATH            Apptainer SIF   [/misc/local/singularity/vllm-openai-v0.26.0.sif]
#   --image REF           Podman image    [docker.io/vllm/vllm-openai:v0.26.0]
#   --venv PATH           Python venv     [$HOME/vllm_env]
#   --gpus N              GPUs to request                     [1]
#   --tp N                tensor-parallel size                [= --gpus]
#   --full-node           reserve the whole node (gpus=8, cpus=96)
#   --queue Q             LSF GPU queue                       [gpu_b300]
#   --gpu-mode M          exclusive_process | shared          [exclusive_process]
#   --cpus N              CPU slots (else gpus * queue-ratio)
#   --port N              server port                         [8000]
#   --max-model-len N     context window                      [32768]
#   --gpu-mem-util F      HBM fraction                        [0.90]
#   --enable-auto-tool-choice     turn on tool calling (needed by chat UIs)
#   --tool-parser NAME            vLLM --tool-call-parser (e.g. kimi_k3)
#   --reasoning-parser NAME       vLLM --reasoning-parser (e.g. kimi_k3)
#   --walltime H:MM       LSF walltime                        [8:00]
#   --host-pin HOST       pin to a node
#   --exit-after-test     stop right after the smoke test
#   --webui               also launch Open WebUI at the model and print its URL
#   --webui-queue Q       CPU queue for the UI                [local]
#   --webui-port N        UI port                             [8080]
#   --domain D            domain suffix for routable URLs     [.int.janelia.org]
#===============================================================================
set -euo pipefail

MODEL=""; SERVED_NAME=""; ENGINE="apptainer"
# Auto-managed (empty = "let the registry/heuristic decide"; an explicit flag wins):
SIF=""; IMAGE=""; QUEUE=""; GPUS=""; FULL_NODE=""; ENABLE_AUTO_TOOL=""
VENV="$HOME/vllm_env"
TP=""; GPU_MODE="exclusive_process"; CPUS=""
PORT=8000; MAX_MODEL_LEN=32768; GPU_MEM_UTIL=0.90; WALLTIME="8:00"
TOOL_PARSER=""; REASONING_PARSER=""
HOST_PIN=""; EXIT_AFTER_TEST=false
WEBUI=false; WEBUI_QUEUE="local"; WEBUI_PORT=8080; DOMAIN=".int.janelia.org"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model=*) MODEL="${1#*=}"; shift ;;               --model) MODEL="$2"; shift 2 ;;
    --served-name=*) SERVED_NAME="${1#*=}"; shift ;;   --served-name) SERVED_NAME="$2"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;             --engine) ENGINE="$2"; shift 2 ;;
    --sif=*) SIF="${1#*=}"; shift ;;                   --sif) SIF="$2"; shift 2 ;;
    --image=*) IMAGE="${1#*=}"; shift ;;               --image) IMAGE="$2"; shift 2 ;;
    --venv=*) VENV="${1#*=}"; shift ;;                 --venv) VENV="$2"; shift 2 ;;
    --gpus=*) GPUS="${1#*=}"; shift ;;                 --gpus) GPUS="$2"; shift 2 ;;
    --tp=*) TP="${1#*=}"; shift ;;                     --tp) TP="$2"; shift 2 ;;
    --full-node) FULL_NODE=true; shift ;;
    --queue=*) QUEUE="${1#*=}"; shift ;;               --queue) QUEUE="$2"; shift 2 ;;
    --gpu-mode=*) GPU_MODE="${1#*=}"; shift ;;         --gpu-mode) GPU_MODE="$2"; shift 2 ;;
    --cpus=*) CPUS="${1#*=}"; shift ;;                 --cpus) CPUS="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;                 --port) PORT="$2"; shift 2 ;;
    --max-model-len=*) MAX_MODEL_LEN="${1#*=}"; shift ;; --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util=*) GPU_MEM_UTIL="${1#*=}"; shift ;; --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    --enable-auto-tool-choice) ENABLE_AUTO_TOOL=true; shift ;;
    --tool-parser=*) TOOL_PARSER="${1#*=}"; shift ;;   --tool-parser) TOOL_PARSER="$2"; shift 2 ;;
    --reasoning-parser=*) REASONING_PARSER="${1#*=}"; shift ;; --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
    --walltime=*) WALLTIME="${1#*=}"; shift ;;         --walltime) WALLTIME="$2"; shift 2 ;;
    --host-pin=*) HOST_PIN="${1#*=}"; shift ;;         --host-pin) HOST_PIN="$2"; shift 2 ;;
    --exit-after-test) EXIT_AFTER_TEST=true; shift ;;
    --webui) WEBUI=true; shift ;;
    --webui-queue=*) WEBUI_QUEUE="${1#*=}"; shift ;;   --webui-queue) WEBUI_QUEUE="$2"; shift 2 ;;
    --webui-port=*) WEBUI_PORT="${1#*=}"; shift ;;     --webui-port) WEBUI_PORT="$2"; shift 2 ;;
    --domain=*) DOMAIN="${1#*=}"; shift ;;             --domain) DOMAIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,58p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

[ -n "$MODEL" ] || { echo "ERROR: --model is required" >&2; exit 1; }
case "$ENGINE" in apptainer|podman|venv) ;; *) echo "ERROR: --engine must be apptainer|podman|venv" >&2; exit 1 ;; esac

# ---- auto-select: registry for known models + size heuristic fallback.
#      Explicit flags ALWAYS win (":=" only fills values you didn't pass). ----
case "$(basename "$MODEL")" in
  Kimi-K3)
      : "${QUEUE:=gpu_b300}"; : "${FULL_NODE:=true}"; : "${SERVED_NAME:=kimi-k3}"
      : "${SIF:=/misc/local/singularity/vllm-kimi-k3.sif}"
      : "${IMAGE:=docker.io/vllm/vllm-openai:kimi-k3}"
      : "${ENABLE_AUTO_TOOL:=true}"; : "${TOOL_PARSER:=kimi_k3}"; : "${REASONING_PARSER:=kimi_k3}" ;;
  Qwen3-32B)
      : "${QUEUE:=gpu_h200}"; : "${GPUS:=1}"
      : "${ENABLE_AUTO_TOOL:=true}"; : "${TOOL_PARSER:=hermes}"; : "${REASONING_PARSER:=qwen3}" ;;
  Qwen3-30B-A3B)
      : "${QUEUE:=gpu_h100}"; : "${GPUS:=1}"
      : "${ENABLE_AUTO_TOOL:=true}"; : "${TOOL_PARSER:=hermes}"; : "${REASONING_PARSER:=qwen3}" ;;
  Qwen2.5-72B*)   # dense 72B (~145 GB bf16) -> TP=2 on H200; instruct, not a reasoning model
      : "${QUEUE:=gpu_h200}"; : "${GPUS:=2}"
      : "${ENABLE_AUTO_TOOL:=true}"; : "${TOOL_PARSER:=hermes}" ;;
  gemma-4-26B*)
      : "${QUEUE:=gpu_h100}"; : "${GPUS:=1}"
      : "${ENABLE_AUTO_TOOL:=true}"; : "${TOOL_PARSER:=gemma4}" ;;
  Mistral-Small-4-119B*)
      : "${QUEUE:=gpu_h200}"; : "${GPUS:=2}"
      : "${ENABLE_AUTO_TOOL:=true}"; : "${TOOL_PARSER:=mistral}" ;;
  gpt-oss-120b)   : "${QUEUE:=gpu_h200}"; : "${GPUS:=2}" ;;   # gpt-oss: harmony format — set parser manually (see guide)
  gpt-oss-20b)    : "${QUEUE:=gpu_h100}"; : "${GPUS:=1}" ;;
esac

# size-heuristic fallback for models not in the registry (weights from safetensors index, else du)
if [ -z "$QUEUE" ]; then
  BYTES=$( { grep -aoE '"total_size"[ :]*[0-9]+' "$MODEL/model.safetensors.index.json" 2>/dev/null | grep -oE '[0-9]+' | tail -1; } || true )
  [ -n "$BYTES" ] || BYTES=$( du -sb "$MODEL" 2>/dev/null | cut -f1 || true )
  HEUR=$(awk -v b="${BYTES:-0}" 'BEGIN{ n=b/1e9*1.3;
     if(n<=22)print "gpu_l4 1"; else if(n<=76)print "gpu_h100 1";
     else if(n<=138)print "gpu_h200 1"; else if(n<=260)print "gpu_b300 1";
     else if(n<=276)print "gpu_h200 2"; else {g=int(n/260); if(g*260<n)g++; print "gpu_b300 " g} }')
  QUEUE="${HEUR%% *}"; : "${GPUS:=${HEUR##* }}"
  echo "[auto] ~$(awk -v b=${BYTES:-0} 'BEGIN{printf "%.0f", b/1e9}') GB weights -> queue=$QUEUE gpus=$GPUS (size heuristic)"
fi

# defaults for anything still unset
[ -n "$SERVED_NAME" ] || SERVED_NAME="$(basename "$MODEL")"
: "${QUEUE:=gpu_b300}"; : "${GPUS:=1}"
: "${SIF:=/misc/local/singularity/vllm-openai-v0.26.0.sif}"
: "${IMAGE:=docker.io/vllm/vllm-openai:v0.26.0}"
FULL_NODE="${FULL_NODE:-false}"; ENABLE_AUTO_TOOL="${ENABLE_AUTO_TOOL:-false}"

# CPU slots per GPU by queue — MUST match the node ratio or bsub stalls.
case "$QUEUE" in *l4*) CPG=8 ;; *) CPG=12 ;; esac
if [ "$FULL_NODE" = true ]; then GPUS=8; fi
[ -n "$TP" ] || TP="$GPUS"
[ -n "$CPUS" ] || CPUS=$((GPUS * CPG))

JOB_NAME="vllm_$(printf '%s' "$SERVED_NAME" | tr -c 'A-Za-z0-9_.-' '_')"
OUTPUT_DIR="$(cd "$(dirname "$0")" && { git rev-parse --show-toplevel 2>/dev/null || pwd; })/output"
mkdir -p "$OUTPUT_DIR"
JOBFILE="$OUTPUT_DIR/.${JOB_NAME}_$$.bsub"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================================"
echo "serve_vllm — $SERVED_NAME  [engine: $ENGINE]"
echo "  model:   $MODEL"
case "$ENGINE" in apptainer) echo "  sif:     $SIF" ;; podman) echo "  image:   $IMAGE" ;; venv) echo "  venv:    $VENV" ;; esac
echo "  queue:   $QUEUE   gpus:$GPUS (tp=$TP) mode=$GPU_MODE cpus=$CPUS full_node=$FULL_NODE"
echo "  serve:   port=$PORT max-model-len=$MAX_MODEL_LEN gpu-mem-util=$GPU_MEM_UTIL walltime=$WALLTIME"
[ "$ENABLE_AUTO_TOOL" = true ] && echo "  tools:   auto-tool-choice on, tool-parser=${TOOL_PARSER:-<none>}, reasoning-parser=${REASONING_PARSER:-<none>}"
[ "$WEBUI" = true ] && echo "  webui:   yes (queue=$WEBUI_QUEUE port=$WEBUI_PORT)"
echo "================================================================"

# ----- vLLM args (base + optional tool/reasoning) -----
VLLM_ARGS="--served-model-name ${SERVED_NAME} --tensor-parallel-size ${TP} --trust-remote-code --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${GPU_MEM_UTIL} --host 0.0.0.0 --port ${PORT}"
[ "$ENABLE_AUTO_TOOL" = true ] && VLLM_ARGS="$VLLM_ARGS --enable-auto-tool-choice"
[ -n "$TOOL_PARSER" ] && VLLM_ARGS="$VLLM_ARGS --tool-call-parser ${TOOL_PARSER}"
[ -n "$REASONING_PARSER" ] && VLLM_ARGS="$VLLM_ARGS --reasoning-parser ${REASONING_PARSER}"

# ----- write the LSF job script to a file (robust + inspectable; auto-removed after submit) -----
{
cat <<HEADER
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -q ${QUEUE}
#BSUB -n ${CPUS}
#BSUB -gpu "num=${GPUS}:mode=${GPU_MODE}"
#BSUB -R "span[hosts=1]"
$( [ -n "$HOST_PIN" ] && echo "#BSUB -m \"$HOST_PIN\"" )
#BSUB -o ${OUTPUT_DIR}/${JOB_NAME}_%J.out
#BSUB -e ${OUTPUT_DIR}/${JOB_NAME}_%J.err
#BSUB -W ${WALLTIME}
HEADER
cat <<'BODY_STATIC'

set -uo pipefail
echo "================================================================"
echo "serve_vllm __SERVED_NAME__ [__ENGINE__] — job $LSB_JOBID on $(hostname -s) — $(date)"
echo "================================================================"
CTR_LOG=__OUTPUT_DIR__/__JOB_NAME___${LSB_JOBID}.vllm.log
CACHE=/scratch/$USER/vllm-cache
mkdir -p "$CACHE"/torchinductor "$CACHE"/triton "$CACHE"/hf "$CACHE"/vllm "$CACHE"/flashinfer
echo "[timing] job_start_epoch=$(date +%s)"
echo "[info] model size: $(du -sh __MODEL__ 2>/dev/null | cut -f1)"

VLLM_ARGS="__VLLM_ARGS__"

if [ "__ENGINE__" = "apptainer" ]; then
    [ -r "__SIF__" ] || { echo "[ERROR] SIF not readable: __SIF__"; exit 1; }
    T_LAUNCH=$(date +%s); echo "[timing] launch_epoch=$T_LAUNCH"
    apptainer exec --nv --cleanenv \
        --bind __MODEL__:/model:ro --bind /scratch/$USER \
        --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,VLLM_LOGGING_LEVEL=INFO,OMP_NUM_THREADS=8 \
        --env TMPDIR=/scratch/$USER,XDG_CACHE_HOME=$CACHE,TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor,TRITON_CACHE_DIR=$CACHE/triton,HF_HOME=$CACHE/hf,VLLM_CACHE_ROOT=$CACHE/vllm,FLASHINFER_CUBIN_DIR=$CACHE/flashinfer \
        "__SIF__" vllm serve /model $VLLM_ARGS > "$CTR_LOG" 2>&1 &
    SRV_PID=$!; ALIVE(){ kill -0 $SRV_PID 2>/dev/null; }; STOP(){ kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null||true; }
    trap STOP EXIT

elif [ "__ENGINE__" = "podman" ]; then
    unset XDG_RUNTIME_DIR
    mkdir -p /scratch/$USER/podman-run /scratch/$USER/podman-storage
    podman system migrate 2>/dev/null || true
    podman info >/dev/null 2>&1 || { echo "[podman] resetting"; podman system reset -f; }
    trap 'podman rm -f __JOB_NAME__ 2>/dev/null; pkill -u $USER catatonit 2>/dev/null' EXIT
    T_LAUNCH=$(date +%s); echo "[timing] launch_epoch=$T_LAUNCH"
    podman pull "__IMAGE__"
    podman run -d --name __JOB_NAME__ --device nvidia.com/gpu=all --ipc=host \
        -v __MODEL__:/model:ro -p __PORT__:__PORT__ \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
        --entrypoint vllm "__IMAGE__" serve /model $VLLM_ARGS
    podman logs -f __JOB_NAME__ > "$CTR_LOG" 2>&1 &
    ALIVE(){ [ "$(podman inspect -f '{{.State.Running}}' __JOB_NAME__ 2>/dev/null)" = "true" ]; }; STOP(){ podman rm -f __JOB_NAME__ 2>/dev/null||true; }

else  # venv
    [ -f "__VENV__/bin/activate" ] || { echo "[ERROR] venv not found: __VENV__"; exit 1; }
    source "__VENV__/bin/activate"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO OMP_NUM_THREADS=8
    export TMPDIR=/scratch/$USER XDG_CACHE_HOME=$CACHE TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor \
           TRITON_CACHE_DIR=$CACHE/triton HF_HOME=$CACHE/hf VLLM_CACHE_ROOT=$CACHE/vllm \
           FLASHINFER_CUBIN_DIR=$CACHE/flashinfer
    T_LAUNCH=$(date +%s); echo "[timing] launch_epoch=$T_LAUNCH"
    vllm serve __MODEL__ $VLLM_ARGS > "$CTR_LOG" 2>&1 &
    SRV_PID=$!; ALIVE(){ kill -0 $SRV_PID 2>/dev/null; }; STOP(){ kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null||true; }
    trap STOP EXIT
fi

echo "[info] waiting for /health on :__PORT__ (log: $CTR_LOG)"
READY=0
for i in $(seq 1 3600); do
    if ! ALIVE; then echo "[ERROR] server exited before ready — tail:"; tail -50 "$CTR_LOG"; exit 1; fi
    if curl -sf "http://127.0.0.1:__PORT__/health" >/dev/null 2>&1; then READY=1; break; fi
    sleep 2
done
T_READY=$(date +%s)
[ $READY -ne 1 ] && { echo "[ERROR] not ready in time"; tail -50 "$CTR_LOG"; exit 1; }

echo "================================================================"
echo "TIMING: launch -> ready = $((T_READY - T_LAUNCH)) s"
grep -iE "loading (model )?weights took|Model loading took" "$CTR_LOG" | head -1
echo "================================================================"
echo "[test] endpoint tests (models / chat / reasoning / tools)"
python3 "__TEST_SCRIPT__" --url "http://127.0.0.1:__PORT__" --model "__SERVED_NAME__" __TOOLS_FLAG__ \
  || echo "[test] endpoint test FAILED (see $CTR_LOG)"

if [ "__EXIT_AFTER_TEST__" = "true" ]; then echo "[info] --exit-after-test; stopping."; STOP
else echo "[info] live at http://$(hostname -s)__DOMAIN__:__PORT__/v1  (model: __SERVED_NAME__)"; echo "[info] bkill $LSB_JOBID to stop."; wait; fi
BODY_STATIC
} > "$JOBFILE"

TEST_SCRIPT="$SCRIPT_DIR/test_vllm_endpoint.py"
TOOLS_FLAG=""; [ "$ENABLE_AUTO_TOOL" = true ] && TOOLS_FLAG="--tools"
sed -i \
  -e "s|__SERVED_NAME__|${SERVED_NAME}|g" -e "s|__ENGINE__|${ENGINE}|g" \
  -e "s|__MODEL__|${MODEL}|g" -e "s|__SIF__|${SIF}|g" -e "s|__IMAGE__|${IMAGE}|g" \
  -e "s|__VENV__|${VENV}|g" -e "s|__PORT__|${PORT}|g" -e "s|__DOMAIN__|${DOMAIN}|g" \
  -e "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" -e "s|__JOB_NAME__|${JOB_NAME}|g" \
  -e "s|__VLLM_ARGS__|${VLLM_ARGS}|g" -e "s|__EXIT_AFTER_TEST__|${EXIT_AFTER_TEST}|g" \
  -e "s|__TEST_SCRIPT__|${TEST_SCRIPT}|g" -e "s|__TOOLS_FLAG__|${TOOLS_FLAG}|g" \
  "$JOBFILE"

SUBMIT=$(bsub < "$JOBFILE"); echo "$SUBMIT"; rm -f "$JOBFILE"
JOBID=$(echo "$SUBMIT" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1)

# ----- optional: launch Open WebUI pointed at wherever the model landed -----
if [ "$WEBUI" = true ] && [ -n "$JOBID" ]; then
    echo -n "Waiting for model job $JOBID to reach a node"
    NODE=""
    for i in $(seq 1 200); do
        st=$(bjobs -o "stat" "$JOBID" 2>/dev/null | tail -1 | tr -d ' ')
        if [ "$st" = "RUN" ]; then NODE=$(bjobs -o "exec_host" "$JOBID" 2>/dev/null | tail -1 | tr -d ' ' | sed 's/^[0-9]*\*//; s/:.*//'); break; fi
        { [ "$st" = "EXIT" ] || [ "$st" = "DONE" ]; } && { echo; echo "model job $st before starting; not launching webui"; exit 1; }
        echo -n "."; sleep 3
    done; echo
    [ -n "$NODE" ] || { echo "model not running yet; launch webui manually once it is:"; echo "  $SCRIPT_DIR/serve_openwebui.sh --model-url http://<node>${DOMAIN}:${PORT}/v1"; exit 0; }
    echo "Model landed on: $NODE (API http://${NODE}${DOMAIN}:${PORT}/v1). Model may still be loading."
    "$SCRIPT_DIR/serve_openwebui.sh" --model-url "http://${NODE}${DOMAIN}:${PORT}/v1" \
        --port "$WEBUI_PORT" --queue "$WEBUI_QUEUE" --domain "$DOMAIN"
fi
