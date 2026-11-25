#!/bin/bash
#===============================================================================
# Universal Ray Cluster Job Submission Script
# Janelia HPC — auto-configures NCCL for IB (B300/H200/H100) or Ethernet (L4/A100)
#
# Usage:
#   ./submit_ray_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]
#
# Examples:
#   # Multi-node on parallel queues (whole nodes, all GPUs, 2+ nodes):
#   ./submit_ray_job.sh 2 --script=test_ray_cluster.py
#   ./submit_ray_job.sh 8 --queue=gpu_b300_parallel --venv=~/ray_env \
#       --script=train_ray.py -- --epochs=50
#
#   # Single-node on non-parallel queues (request specific GPU count):
#   ./submit_ray_job.sh 1 --queue=gpu_h100 --num-gpus=2 --venv=~/ray_env \
#       --script=train_ray.py -- --epochs=20
#
# Parallel queues (whole nodes, 2 or more nodes — never single-node jobs):
#   gpu_b300_parallel  -> 96 CPUs, 8 GPUs, InfiniBand NCCL
#   gpu_h200_parallel  -> 96 CPUs, 8 GPUs, InfiniBand NCCL
#   gpu_h100_parallel  -> 96 CPUs, 8 GPUs, InfiniBand NCCL  [default]
#   gpu_l4_parallel    -> 64 CPUs, 8 GPUs, Ethernet NCCL
#   gpu_a100_parallel  -> 48 CPUs, 4 GPUs, Ethernet NCCL
#   gpu_rtx6000_parallel -> 128 CPUs, 8 GPUs, Ethernet NCCL
#
# Non-parallel queues (single node, flexible GPU count):
#   gpu_b300           -> use --num-gpus=N (1-8), InfiniBand NCCL
#   gpu_h200           -> use --num-gpus=N (1-8), InfiniBand NCCL
#   gpu_h100           -> use --num-gpus=N (1-8), InfiniBand NCCL
#   gpu_l4             -> use --num-gpus=N (1-8), Ethernet NCCL
#   gpu_l4_16          -> use --num-gpus=N (1-8), Ethernet NCCL
#   gpu_l4_large       -> 1 GPU, Ethernet NCCL
#   gpu_rtx6000        -> use --num-gpus=N (1-8), Ethernet NCCL
#   gpu_a100           -> use --num-gpus=N (1-4), Ethernet NCCL
# Default CPUs per GPU on non-parallel queues follow the site slot ratio:
#   l4=8, l4_16=16, l4_large=64, rtx6000=16, a100/h100/h200/b300=12
#===============================================================================

usage() {
    echo "Usage: $0 <num_nodes> --script=SCRIPT [options] [-- script_args...]"
    echo ""
    echo "Required:"
    echo "  <num_nodes>          Number of nodes"
    echo "  --script=FILE        Python script to run"
    echo ""
    echo "Optional:"
    echo "  --queue=QUEUE        LSF queue (default: gpu_h100_parallel)"
    echo "  --num-gpus=N         GPUs to request (non-parallel queues, default: 1)"
    echo "  --num-cpus=N         CPUs to request (non-parallel queues, default: queue's slots per GPU)"
    echo "  --job-name=NAME      Job name (default: ray_job)"
    echo "  --walltime=TIME      Walltime in hours (default: 24:00 non-parallel, auto parallel)"
    echo "  --venv=PATH          Python venv path"
    echo "  --sandbox=KIND       Run Ray and the script in one sandbox per node: bwrap or podman"
    echo "                       (agentic-sandbox). The network is chosen by queue: InfiniBand queues"
    echo "                       reach the job's own nodes only; Ethernet queues keep the node's network."
    echo "  --sandbox-dir=PATH   Where sandbox-run.sh / podman-run.sh live (default: \$AGENTIC_SANDBOX_DIR,"
    echo "                       else wherever they are found on PATH)"
    echo "  --image=IMAGE        Container image for --sandbox=podman (default: agentic-sandbox-gpu)"
    echo "  --ro=PATH, --rw=PATH Extra directory the sandbox may read / write (repeatable): datasets,"
    echo "                       and where the script writes (e.g. --rw=/scratch/\$USER/ray_results)"
    echo ""
    echo "Script arguments: use '--' separator"
    echo "  Example: $0 2 --script=train.py -- --epochs=50 --batch-size=128"
    echo "  Example: $0 1 --queue=gpu_h100 --num-gpus=2 --script=train.py -- --epochs=10"
    exit 1
}

if [ $# -lt 2 ]; then usage; fi

NUM_NODES=$1
shift

QUEUE_NAME="gpu_h100_parallel"
PYTHON_SCRIPT=""
JOB_NAME="ray_job"
WALLTIME=""
VENV_PATH=""
SCRIPT_ARGS=""
USER_NUM_GPUS=""
USER_NUM_CPUS=""
SANDBOX=""
SANDBOX_DIR="${AGENTIC_SANDBOX_DIR:-}"
SANDBOX_IMAGE="ghcr.io/janeliascientificcomputingsystems/agentic-sandbox-gpu:latest"
SANDBOX_RO=()
SANDBOX_RW=()

# bash does not expand "~" inside --opt=~/path, so do it for the sandbox's path options
expand_tilde() {
    case "$1" in
        "~")    echo "$HOME" ;;
        "~/"*)  echo "$HOME/${1#\~/}" ;;
        *)      echo "$1" ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue=*)    QUEUE_NAME="${1#*=}"; shift ;;
        --script=*)   PYTHON_SCRIPT="${1#*=}"; shift ;;
        --num-gpus=*) USER_NUM_GPUS="${1#*=}"; shift ;;
        --num-cpus=*) USER_NUM_CPUS="${1#*=}"; shift ;;
        --job-name=*) JOB_NAME="${1#*=}"; shift ;;
        --walltime=*) WALLTIME="${1#*=}"; shift ;;
        --venv=*)     VENV_PATH="${1#*=}"; shift ;;
        --sandbox=*)     SANDBOX="${1#*=}"; shift ;;
        --sandbox-dir=*) SANDBOX_DIR="$(expand_tilde "${1#*=}")"; shift ;;
        --image=*)       SANDBOX_IMAGE="${1#*=}"; shift ;;
        --ro=*)          SANDBOX_RO+=("$(expand_tilde "${1#*=}")"); shift ;;
        --rw=*)          SANDBOX_RW+=("$(expand_tilde "${1#*=}")"); shift ;;
        --) shift; SCRIPT_ARGS=$(printf '%q ' "$@"); break ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$PYTHON_SCRIPT" ]; then echo "ERROR: --script is required"; usage; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "ERROR: Script not found: $PYTHON_SCRIPT"; exit 1; fi

# Auto-configure resources and network backend per queue
PARALLEL_QUEUE=true
case $QUEUE_NAME in
    # Parallel queues — whole nodes, multi-node capable
    gpu_b300_parallel|gpu_h200_parallel|gpu_h100_parallel)
        CPUS_PER_NODE=96
        GPUS_PER_NODE=8
        APP_PROFILE="parallel-96"
        NETWORK_BACKEND="IB"
        ;;
    gpu_l4_parallel)
        CPUS_PER_NODE=64
        GPUS_PER_NODE=8
        APP_PROFILE="parallel-64"
        NETWORK_BACKEND="ETH"
        ;;
    gpu_a100_parallel)
        CPUS_PER_NODE=48
        GPUS_PER_NODE=4
        APP_PROFILE="parallel-48"
        NETWORK_BACKEND="ETH"
        ;;
    gpu_rtx6000_parallel)
        CPUS_PER_NODE=128
        GPUS_PER_NODE=8
        APP_PROFILE="parallel-128"
        NETWORK_BACKEND="ETH"
        ;;
    # Non-parallel queues — single node, flexible GPU count
    # Defaults: 1 GPU, the queue's slots-per-GPU ratio, 24h walltime
    gpu_b300|gpu_h200|gpu_h100)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 12))}
        APP_PROFILE=""
        NETWORK_BACKEND="IB"
        ;;
    gpu_l4)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 8))}
        APP_PROFILE=""
        NETWORK_BACKEND="ETH"
        ;;
    gpu_l4_16|gpu_rtx6000)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 16))}
        APP_PROFILE=""
        NETWORK_BACKEND="ETH"
        ;;
    gpu_l4_large)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 64))}
        APP_PROFILE=""
        NETWORK_BACKEND="ETH"
        ;;
    gpu_a100)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 12))}
        APP_PROFILE=""
        NETWORK_BACKEND="ETH"
        ;;
    *)
        echo "ERROR: Unsupported queue: $QUEUE_NAME"
        echo "Parallel:     gpu_b300_parallel, gpu_h200_parallel, gpu_h100_parallel, gpu_l4_parallel, gpu_a100_parallel, gpu_rtx6000_parallel"
        echo "Non-parallel: gpu_b300, gpu_h200, gpu_h100, gpu_l4, gpu_l4_16, gpu_l4_large, gpu_rtx6000, gpu_a100 (use --num-gpus=N)"
        exit 1
        ;;
esac

if [ "$PARALLEL_QUEUE" = false ] && [ $NUM_NODES -gt 1 ]; then
    echo "ERROR: Non-parallel queues only support 1 node. Use a parallel queue for multi-node."
    exit 1
fi

# Site policy: *_parallel queues are for multi-node jobs only (they allocate
# whole nodes without CPU fencing or memory limits).
if [ "$PARALLEL_QUEUE" = true ] && [ $NUM_NODES -lt 2 ]; then
    echo "ERROR: $QUEUE_NAME is a parallel queue — never submit single-node jobs to it."
    echo "       For one node use ${QUEUE_NAME%_parallel} with --num-gpus=N (up to ${GPUS_PER_NODE})."
    exit 1
fi

TOTAL_CPUS=$((NUM_NODES * CPUS_PER_NODE))
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Default walltime: parallel queues auto-scale, non-parallel 24h
if [ -z "$WALLTIME" ]; then
    if [ "$PARALLEL_QUEUE" = true ]; then
        [ $NUM_NODES -le 2 ] && WALLTIME="4:00" || WALLTIME="8:00"
    else
        WALLTIME="24:00"
    fi
fi
echo "================================================================"
echo "Ray Cluster Job Submission"
echo "================================================================"
echo "Script:          $PYTHON_SCRIPT"
echo "Script args:     $SCRIPT_ARGS"
echo "Nodes:           $NUM_NODES"
echo "Queue:           $QUEUE_NAME"
echo "Network backend: $NETWORK_BACKEND"
echo "Total GPUs:      $TOTAL_GPUS"
echo "Total CPUs:      $TOTAL_CPUS"
echo "Walltime:        $WALLTIME"
echo "================================================================"

APP_LINE=""
SPAN_LINE=""
if [ -n "$APP_PROFILE" ]; then
    APP_LINE="#BSUB -app ${APP_PROFILE}"
fi
if [ "$PARALLEL_QUEUE" = true ]; then
    SPAN_LINE="#BSUB -R \"span[ptile=${CPUS_PER_NODE}]\""
    # Whole nodes are ours alone: safe to clear any stale Ray daemon left on
    # the node by a prior job before starting.
    STALE_STOP="ray stop --force >/dev/null 2>&1 || true"
else
    # Single-node job on a shared node: all slots must be on one host, and
    # other Ray clusters on that host (ours or not) must be left alone.
    SPAN_LINE="#BSUB -R \"span[hosts=1]\""
    STALE_STOP=":"
fi

# NCCL env file path (for IB queues)
case $QUEUE_NAME in
    gpu_b300_parallel|gpu_b300) NCCL_ENV_FILE="$(cd "$(dirname "$0")" && pwd)/../nccl-ib/nccl_ib_b300.sh" ;;
    *)                          NCCL_ENV_FILE="$(cd "$(dirname "$0")" && pwd)/../nccl-ib/nccl_ib.sh" ;;
esac

# LSF does not create the log directory; without it the job output is lost
mkdir -p ../output

#===============================================================================
# --sandbox: Ray and the script run inside one agentic-sandbox sandbox per node
# (sandbox-run.sh = Bubblewrap, podman-run.sh = a container). This launcher and the job
# script stay outside. Use it for code you did not write yourself, such as code an AI agent
# wrote: the sandbox sees the submission directory, ../output and what --ro/--rw name, and
# nothing else of your account (no home directory, keys or tokens).
#
# The shape differs from the unsandboxed path below, where 'ray start' leaves daemons on each
# node and the script runs in the job script. A sandbox ends with its command, so:
#   head node     one sandbox: ray start --head, wait for the nodes, run the script, ray stop
#   other nodes   one sandbox each: ray start --address, then stay until the raylet exits
#
# The network is chosen here, never by the user:
#   two or more nodes  --host-net : the node's own network, so that the nodes reach each other.
#                      On InfiniBand queues --ib as well: NCCL's data goes over InfiniBand
#   one node           no network: Ray runs on the sandbox's own loopback
# Every Ray port is fixed. A multi-node job has its nodes to itself on the parallel queues, and
# a one-node sandbox has its own network namespace, so the ports cannot collide with another job.
#===============================================================================
if [ -n "$SANDBOX" ]; then
    case "$SANDBOX" in
        bwrap)  SANDBOX_WRAPPER="sandbox-run.sh" ;;
        podman) SANDBOX_WRAPPER="podman-run.sh" ;;
        *) echo "ERROR: --sandbox takes bwrap or podman (got '$SANDBOX')"; exit 1 ;;
    esac
    if [ -z "$SANDBOX_DIR" ]; then
        found=$(command -v "$SANDBOX_WRAPPER" 2>/dev/null) && SANDBOX_DIR=$(dirname "$(readlink -f "$found")")
    fi
    [ -d "$SANDBOX_DIR" ] && SANDBOX_DIR=$(cd "$SANDBOX_DIR" && pwd)
    if [ ! -x "$SANDBOX_DIR/$SANDBOX_WRAPPER" ]; then
        echo "ERROR: $SANDBOX_WRAPPER not found. Pass --sandbox-dir=PATH (the scripts/ directory of"
        echo "       https://github.com/JaneliaScientificComputingSystems/agentic-sandbox) or set AGENTIC_SANDBOX_DIR."
        exit 1
    fi
    VENV_ABS=""
    if [ -n "$VENV_PATH" ]; then
        if [ "$SANDBOX" = podman ]; then
            echo "ERROR: --sandbox=podman runs the image's own Python and Ray; a host venv does not work inside it."
            echo "       Drop --venv, or choose an image that has what the script needs (--image=...)."
            exit 1
        fi
        VENV_ABS=$(cd "$(expand_tilde "$VENV_PATH")" 2>/dev/null && pwd) || { echo "ERROR: --venv $VENV_PATH: no such directory"; exit 1; }
    fi
    SUBMIT_DIR=$(pwd -P)
    OUT_ABS=$(cd ../output && pwd -P)
    SCRIPT_ABS=$(cd "$(dirname "$PYTHON_SCRIPT")" && pwd -P)/$(basename "$PYTHON_SCRIPT")
    # Canonical (no ".."): the path is compared with the mounts below and used inside the sandbox.
    [ -f "$NCCL_ENV_FILE" ] && NCCL_ENV_FILE=$(cd "$(dirname "$NCCL_ENV_FILE")" && pwd -P)/$(basename "$NCCL_ENV_FILE")

    # Mounts. Bubblewrap binds the submission directory read-write by itself; podman mounts only
    # what it is told. A path already inside a read-write mount is not added again. The --rw
    # directories are created on every node first (node-local /scratch paths do not exist yet).
    SB_MOUNTS=(); SB_RW=(); SB_MKDIR=()
    sb_covered() { local q; for q in ${SB_RW[@]+"${SB_RW[@]}"}; do [ "$1" = "$q" ] || [ "${1#"$q"/}" != "$1" ] && return 0; done; return 1; }
    sb_add() {  # sb_add ro|rw PATH
        sb_covered "$2" && return 0
        SB_MOUNTS+=("--$1" "$2"); [ "$1" = rw ] && SB_RW+=("$2")
        return 0
    }
    if [ "$SANDBOX" = podman ]; then sb_add rw "$SUBMIT_DIR"; else SB_RW+=("$SUBMIT_DIR"); fi
    sb_add rw "$OUT_ABS"
    for d in ${SANDBOX_RW[@]+"${SANDBOX_RW[@]}"}; do
        case "$d" in /*) ;; *) echo "ERROR: --rw=$d: give an absolute path"; exit 1 ;; esac
        SB_MKDIR+=("$d"); sb_add rw "$d"
    done
    sb_add ro "$(dirname "$SCRIPT_ABS")"
    [ -n "$VENV_ABS" ] && sb_add ro "$VENV_ABS"
    [ "$NETWORK_BACKEND" = "IB" ] && [ -f "$NCCL_ENV_FILE" ] && sb_add ro "$(dirname "$NCCL_ENV_FILE")"
    for d in ${SANDBOX_RO[@]+"${SANDBOX_RO[@]}"}; do
        d=$(cd "$d" 2>/dev/null && pwd -P) || { echo "ERROR: --ro: no such directory"; exit 1; }
        sb_add ro "$d"
    done
    SB_EXTRA=()
    PREPULL_IMAGE=""
    if [ "$SANDBOX" = podman ]; then
        # Ray keeps its object store in /dev/shm; the size is a cap, not a reservation.
        SB_EXTRA=(--image "$SANDBOX_IMAGE" --shm-size 64g); PREPULL_IMAGE="$SANDBOX_IMAGE"
    fi

    SB_INNER="$OUT_ABS/.ray_sandbox_inner_$$.sh"
    SB_NODE="$OUT_ABS/.ray_sandbox_node_$$.sh"

    # ---- runs INSIDE the sandbox, once per node: ROLE MY_ADDR HEAD_ADDR NUM_NODES
    {
        echo '#!/bin/bash'
        echo 'ROLE=$1 MYIP=$2 HEAD=$3 NODES=$4 JOB=$5'
        echo "export USER=$(printf '%q' "$USER") LOGNAME=$(printf '%q' "$USER") PYTHONUNBUFFERED=1 RAY_USAGE_STATS_ENABLED=0"
        [ -n "$VENV_ABS" ] && echo "source $(printf '%q' "$VENV_ABS/bin/activate")"
        if [ "$NETWORK_BACKEND" = "IB" ]; then
            echo "source $(printf '%q' "$NCCL_ENV_FILE") || exit 1"
            # One node: the sandbox has no network, only its own loopback, and the interface
            # the settings name does not exist in it. NCCL then finds the loopback by itself.
            echo '[ "$NODES" -eq 1 ] && unset NCCL_SOCKET_IFNAME'
        else
            echo 'export NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=0 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_BUFFSIZE=8388608 NCCL_DEBUG=INFO'
        fi
        echo "CPUS=${CPUS_PER_NODE} GPUS=${GPUS_PER_NODE}"
        echo "cd $(printf '%q' "$SUBMIT_DIR") || exit 1"
        cat << 'INNER_EOF'
h=$(hostname -s)
PY=$(command -v python || command -v python3)
# Every port another node connects to is fixed, from 11000 up: below the range the system
# picks its own ports from, and clear of Ray's own default for its client server (10001).
B=11000
GCS=$((B + 1))
# The head writes its port here once Ray is up; the other nodes wait for the file.
GCS_FILE=.ray_gcs_$JOB
COMMON=(--node-ip-address="$MYIP" --num-cpus="$CPUS" --num-gpus="$GPUS"
        --node-manager-port=$((B + 2)) --object-manager-port=$((B + 3))
        --dashboard-agent-grpc-port=$((B + 5)) --dashboard-agent-listen-port=$((B + 6))
        --metrics-export-port=$((B + 7)) --runtime-env-agent-port=$((B + 8))
        --min-worker-port=$((B + 100)) --max-worker-port=$((B + 499)))
nodes_up() {
    timeout 30 ray status --address "$HEAD:$GCS" 2>/dev/null | awk '
        /^(Active|Healthy):/ { a = 1; next } /^[A-Za-z]/ { a = 0 }
        a && $1 ~ /^[0-9]+$/ { n += $1 } END { print n + 0 }'
}
echo "[sandbox $h] role=$ROLE address=$MYIP gpus=$(nvidia-smi -L 2>/dev/null | wc -l) $(ray --version 2>&1 | tail -1)"
if [ "$ROLE" = head ]; then
    ray start --head --port="$GCS" --dashboard-port=$((B + 4)) --temp-dir=/tmp/ray "${COMMON[@]}" \
        || { echo "ERROR: 'ray start --head' failed on $h"; exit 1; }
    echo "$GCS" > "$GCS_FILE.tmp" && mv "$GCS_FILE.tmp" "$GCS_FILE"
    n=0
    for i in $(seq 1 100); do
        n=$(nodes_up); [ "$n" -ge "$NODES" ] && break
        echo "Waiting for nodes to join: $n/$NODES"; sleep 3
    done
    if [ "$n" -lt "$NODES" ]; then
        echo "ERROR: only $n of $NODES nodes joined the Ray cluster"
        ray status --address "$HEAD:$GCS"; ray stop >/dev/null 2>&1; exit 1
    fi
    echo "================================================================"
    echo "Ray cluster ready: all $n nodes joined"
    ray status --address "$HEAD:$GCS"
    echo "================================================================"
    RAY_ADDRESS="$HEAD:$GCS" "$PY" __SCRIPT_CMD__
    rc=$?
    echo "Script exited: $rc"
    ray stop >/dev/null 2>&1
    exit $rc
else
    for i in $(seq 1 150); do [ -s "$GCS_FILE" ] && break; sleep 2; done
    [ -s "$GCS_FILE" ] || { echo "ERROR: $h: the Ray head did not start within 5 minutes"; exit 1; }
    GCS=$(cat "$GCS_FILE")
    joined=0
    for i in $(seq 1 60); do
        ray start --address="$HEAD:$GCS" "${COMMON[@]}" && { joined=1; break; }
        echo "[sandbox $h] head not reachable yet, retrying"; sleep 5
    done
    [ $joined -eq 1 ] || { echo "ERROR: $h could not join the Ray head at $HEAD:$GCS"; exit 1; }
    echo "[sandbox $h] joined the cluster"
    # Stay for as long as this node's raylet lives: it exits when the head stops Ray.
    while pgrep -f raylet >/dev/null; do sleep 5; done
    echo "[sandbox $h] Ray stopped, leaving"
fi
INNER_EOF
    } > "$SB_INNER"
    python3 -c "
import sys
p, cmd = sys.argv[1], sys.argv[2]
s = open(p).read()
open(p, 'w').write(s.replace('__SCRIPT_CMD__', cmd))
" "$SB_INNER" "$(printf '%q' "$SCRIPT_ABS") ${SCRIPT_ARGS}"

    # ---- runs on each node OUTSIDE the sandbox, started by blaunch: ROLE HEAD_ADDR NUM_NODES
    {
        echo '#!/bin/bash'
        echo 'ROLE=$1 HEAD=$2 NODES=$3'
        echo "cd $(printf '%q' "$SUBMIT_DIR") || exit 1"
        for d in ${SB_MKDIR[@]+"${SB_MKDIR[@]}"}; do echo "mkdir -p $(printf '%q' "$d")"; done
        echo 'NET=(); MYIP=127.0.0.1'
        echo 'if [ "$NODES" -gt 1 ]; then'
        echo "    MYIP=\$(getent ahostsv4 \"\$(hostname)\" | awk 'NR==1{print \$1}')"
        if [ "$NETWORK_BACKEND" = "IB" ]; then
            echo '    NET=(--host-net --ib)'
            echo 'else'
            echo '    NET=(--ib)'
        else
            echo '    NET=(--host-net)'
        fi
        echo 'fi'
        printf 'exec %q --gpu "${NET[@]}"' "$SANDBOX_DIR/$SANDBOX_WRAPPER"
        for a in ${SB_EXTRA[@]+"${SB_EXTRA[@]}"} ${SB_MOUNTS[@]+"${SB_MOUNTS[@]}"}; do printf ' %q' "$a"; done
        printf ' -- %q "$ROLE" "$MYIP" "$HEAD" "$NODES" "$LSB_JOBID"\n' "$SB_INNER"
    } > "$SB_NODE"
    chmod +x "$SB_INNER" "$SB_NODE"

    SB_RO_LIST=""; for ((k = 0; k < ${#SB_MOUNTS[@]}; k += 2)); do [ "${SB_MOUNTS[k]}" = --ro ] && SB_RO_LIST+=" ${SB_MOUNTS[k+1]}"; done
    echo "Sandbox:         $SANDBOX ($SANDBOX_DIR/$SANDBOX_WRAPPER)"
    if [ "$NUM_NODES" -eq 1 ]; then echo "Sandbox net:     none (one node)"
    elif [ "$NETWORK_BACKEND" = IB ]; then echo "Sandbox net:     the job's own nodes only, data over InfiniBand"
    else echo "Sandbox net:     the node's own network (Ethernet queue)"; fi
    echo "Sandbox write:   ${SB_RW[*]}"
    echo "Sandbox read:   ${SB_RO_LIST:- (nothing else)}"
    echo "================================================================"

    cat << EOF | bsub
#!/bin/bash
#BSUB -J ${JOB_NAME}_${NUM_NODES}nodes
#BSUB -n ${TOTAL_CPUS}
${APP_LINE}
#BSUB -q ${QUEUE_NAME}
#BSUB -gpu "num=${GPUS_PER_NODE}:mode=exclusive_process"
${SPAN_LINE}
#BSUB -o ../output/${JOB_NAME}_%J.out
#BSUB -e ../output/${JOB_NAME}_%J.err
#BSUB -W ${WALLTIME}

hosts=()
for host in \$(cat \$LSB_DJOB_HOSTFILE | uniq); do hosts+=(\$host); done
NODES=\${#hosts[@]}
head_node=\${hosts[0]}
HEAD_ADDR=127.0.0.1
[ \$NODES -gt 1 ] && HEAD_ADDR=\$(getent ahostsv4 \$head_node | awk 'NR==1{print \$1}')
echo "Host list: \${hosts[@]}"
echo "Nodes: \$NODES | GPUs: ${TOTAL_GPUS} | Network: ${NETWORK_BACKEND} | Sandbox: ${SANDBOX}"
echo "Head node: \$head_node (\$HEAD_ADDR)"

trap 'rm -f $(printf '%q' "$SB_INNER") $(printf '%q' "$SB_NODE") $(printf '%q' "$SUBMIT_DIR")/.ray_gcs_\$LSB_JOBID' EXIT
trap 'echo "Job interrupted by signal"; exit 143' INT TERM USR2

# --sandbox=podman: every node keeps its own image store; a node that still has to pull would
# join minutes late. Pull everywhere first.
if [ -n "${PREPULL_IMAGE}" ]; then
    echo "Pulling ${PREPULL_IMAGE} on every node..."
    for host in "\${hosts[@]}"; do
        blaunch -z \$host podman --events-backend=none pull -q ${PREPULL_IMAGE} > /dev/null 2>&1 &
    done
    wait
    echo "Image pulled: \$(date)"
fi

echo ""
echo "================================================================"
echo "Starting Ray, one sandbox per node..."
echo "================================================================"
if [ \$NODES -eq 1 ]; then
    $(printf '%q' "$SB_NODE") head \$HEAD_ADDR 1
    script_exit_code=\$?
else
    blaunch -z \$head_node $(printf '%q' "$SB_NODE") head \$HEAD_ADDR \$NODES &
    head_pid=\$!
    worker_pids=()
    for host in "\${hosts[@]:1}"; do
        blaunch -z \$host $(printf '%q' "$SB_NODE") worker \$HEAD_ADDR \$NODES &
        worker_pids+=(\$!)
    done
    wait \$head_pid
    script_exit_code=\$?
    # The head has stopped Ray; the other sandboxes leave by themselves once their raylet is
    # gone. Give them a minute, then end whatever is left.
    for i in \$(seq 1 12); do
        kill -0 "\${worker_pids[@]}" 2>/dev/null || break
        sleep 5
    done
    kill "\${worker_pids[@]}" 2>/dev/null
    wait
fi
echo "Ray cluster stopped"

if [ \$script_exit_code -ne 0 ]; then
    echo "================================================================"
    echo "JOB FAILED — exit code: \$script_exit_code"
    echo "================================================================"
    exit \$script_exit_code
else
    echo "================================================================"
    echo "JOB COMPLETED SUCCESSFULLY"
    echo "================================================================"
fi
EOF
    exit $?
fi

cat << EOF | bsub
#!/bin/bash
#BSUB -J ${JOB_NAME}_${NUM_NODES}nodes
#BSUB -n ${TOTAL_CPUS}
${APP_LINE}
#BSUB -q ${QUEUE_NAME}
#BSUB -gpu "num=${GPUS_PER_NODE}:mode=exclusive_process"
${SPAN_LINE}
#BSUB -o ../output/${JOB_NAME}_%J.out
#BSUB -e ../output/${JOB_NAME}_%J.err
#BSUB -W ${WALLTIME}

if [ -n "${VENV_PATH}" ]; then source ${VENV_PATH}/bin/activate; fi

# Unbuffered Python stdout/stderr so per-epoch training prints appear in the
# .out log in real time instead of being lost in block buffers on exit.
export PYTHONUNBUFFERED=1

#-----------------------------------------------------------------------
# NCCL configuration — sourced from nccl_ib.sh for IB queues
# All NCCL env vars live in one place to avoid duplication
#-----------------------------------------------------------------------
# GPU indices LSF assigned to this job (empty if LSF set none)
LSF_CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-}"

if [ "${NETWORK_BACKEND}" = "IB" ]; then
    echo "NCCL backend: InfiniBand"
    NCCL_ENV_FILE="${NCCL_ENV_FILE}"
    source "\$NCCL_ENV_FILE"
    echo "  HCAs: \$NCCL_IB_HCA"
else
    NCCL_ENV_FILE=""
    echo "NCCL backend: Ethernet"
    export NCCL_IB_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0
    export NCCL_BUFFSIZE=8388608
    export NCCL_DEBUG=INFO
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

# LSF sets CUDA_VISIBLE_DEVICES to the GPUs it assigned; site policy is not to
# override it, so its value replaces the 0-7 set above on every queue. Only if
# LSF set none: whole-node (parallel) jobs keep 0-7, shared-node jobs get none.
# GPU_ENV_FIX is re-applied in the ray start shells after they source the
# NCCL env file.
GPU_ENV_FIX=":"
if [ -n "\$LSF_CUDA_VISIBLE_DEVICES" ]; then
    GPU_ENV_FIX="export CUDA_VISIBLE_DEVICES=\$LSF_CUDA_VISIBLE_DEVICES"
elif [ "${PARALLEL_QUEUE}" != true ]; then
    GPU_ENV_FIX="unset CUDA_VISIBLE_DEVICES"
fi
\$GPU_ENV_FIX

# Per-job temp dir so concurrent/back-to-back jobs never share Ray session state
export RAY_TMPDIR="/scratch/\$(whoami)/ray_\$(whoami)_\${LSB_JOBID}"
mkdir -p \$RAY_TMPDIR

function getfreeport() {
    CHECK="do while"
    while [[ ! -z \$CHECK ]]; do
        port=\$(( ( RANDOM % 28000 ) + 32000 ))
        CHECK=\$(ss -tln | grep ":\$port ")
    done
    echo \$port
}

# Tag exported only in the 'ray start' shells. Every Ray daemon of this job
# inherits it, which is how stop_job_ray finds them without touching any other
# Ray cluster the same user has on a shared node.
RAY_JOB_TAG="lsf_\${LSB_JOBID}"

# Stop the Ray processes of ONE job on the local host and remove its temp dir.
# Matches the per-job temp dir in the command line (gcs_server, raylet,
# dashboard, agents, log monitor) or the job tag in the process environment;
# Ray worker processes exit on their own once their raylet is gone.
# Shipped to each host with 'declare -f' — a plain 'ray stop' would kill every
# Ray process the user owns on that host, including other jobs' clusters.
function stop_job_ray() {
    local tmpdir=\$1 tag=\$2 pids sig i
    for sig in TERM KILL; do
        pids=\$( { pgrep -u \$(id -u) -f "\$tmpdir/sessio[n]_"
                  grep -lsz "^LSF_RAY_JOB_TAG=\$tag\\\$" /proc/[0-9]*/environ | cut -d/ -f3
                } | sort -u )
        [ -z "\$pids" ] && break
        kill -\$sig \$pids 2>/dev/null
        if [ \$sig = TERM ]; then
            for i in 1 2 3 4 5 6 7 8 9 10; do
                kill -0 \$pids 2>/dev/null || break
                sleep 1
            done
        fi
    done
    rm -rf "\$tmpdir"
}

# Per-job Ray ports, identical on every host of the job, so two clusters on a
# shared node do not collide on fixed defaults. A block of 10 ports in
# 20000-31999 is derived from the job ID and moved on if busy on this host.
# (The head/GCS port from getfreeport is drawn from 32000-59999.)
function ports_free() {
    local p
    for p in "\$@"; do
        ss -tln 2>/dev/null | grep -q ":\$p " && return 1
    done
    return 0
}
port_block=\$(( \${LSB_JOBID:-0} % 1200 ))
for try in 1 2 3 4 5 6 7 8 9 10; do
    port_base=\$(( 20000 + port_block * 10 ))
    ports_free \$((port_base + 1)) \$((port_base + 2)) \$((port_base + 3)) \$((port_base + 4)) && break
    port_block=\$(( (port_block + 7) % 1200 ))
done
agent_grpc_port=\$((port_base + 1))
agent_listen_port=\$((port_base + 2))
dashboard_port=\$((port_base + 3))
# The Ray Client server only starts when its dependencies are installed
# (ray[client]); 'ray start' rejects the port flag without them. When they are,
# move it off the fixed default 10001 so two clusters on one host do not collide.
client_flag=""
python -c 'import grpc' 2>/dev/null && client_flag="--ray-client-server-port=\$((port_base + 4))"

# Worker ports. Every Ray cluster hands its workers ports from 10002-19999 by
# default, so two clusters starting on one shared node give out the same ports:
# the workers crash on bind and the driver hangs in ray.init(). Each job gets
# its own range of 500 ports in 10000-19999, moved on if anything on this host
# already listens inside it.
function range_free() {
    ss -tln 2>/dev/null | awk -v lo=\$1 -v hi=\$2 '
        { n = split(\$4, a, ":"); p = a[n] + 0; if (p >= lo && p <= hi) busy = 1 }
        END { exit busy }'
}
# Jobs that start in the same second cannot see each other's listeners yet, so
# a range is also claimed with an atomic mkdir on this host. A claim only has
# to cover the startup window: one older than 3 minutes is taken over.
port_claims=/scratch/\$(whoami)/.ray_port_claims
port_claim=""
function claim_slot() {
    local d=\$port_claims/slot_\$1
    mkdir -p \$port_claims 2>/dev/null
    if mkdir \$d 2>/dev/null; then port_claim=\$d; return 0; fi
    if [ -n "\$(find \$d -maxdepth 0 -mmin +3 2>/dev/null)" ]; then
        touch \$d; port_claim=\$d; return 0
    fi
    return 1
}
worker_slot=\$(( \${LSB_JOBID:-0} % 20 ))
for try in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    min_worker_port=\$(( 10000 + worker_slot * 500 ))
    max_worker_port=\$(( min_worker_port + 499 ))
    range_free \$min_worker_port \$max_worker_port && claim_slot \$worker_slot && break
    worker_slot=\$(( (worker_slot + 1) % 20 ))
done

hosts=()
for host in \$(cat \$LSB_DJOB_HOSTFILE | uniq); do
    echo "Adding host: \$host"
    hosts+=(\$host)
done
echo "Host list: \${hosts[@]}"
echo "Nodes: ${NUM_NODES} | GPUs: ${TOTAL_GPUS} | Network: ${NETWORK_BACKEND}"

for host in "\${hosts[@]}"; do
    blaunch -z \$host "mkdir -p \$RAY_TMPDIR" &
done
wait

# Stop Ray and remove the per-job temp dir on every host. Runs on normal exit,
# on startup failure, and when LSF signals the job (bkill / walltime).
cleanup_done=0
function cleanup() {
    [ \$cleanup_done -eq 1 ] && return
    cleanup_done=1
    [ -n "\$port_claim" ] && rmdir \$port_claim 2>/dev/null
    for host in "\${hosts[@]}"; do
        blaunch -z \$host "\$(declare -f stop_job_ray); stop_job_ray '\$RAY_TMPDIR' '\$RAY_JOB_TAG'" &
    done
    wait
    echo "Ray cluster stopped"
}
trap cleanup EXIT
trap 'echo "Job interrupted by signal — cleaning up"; exit 143' INT TERM USR2

# Seconds to wait for the head / each worker / the full node count
RAY_START_TIMEOUT=300

# IB pre-flight check (IB queues only)
if [ "${NETWORK_BACKEND}" = "IB" ]; then
    echo ""
    echo "Running IB pre-flight check..."
    blaunch -z \${hosts[0]} "ibv_devinfo | grep hca_id" || {
        echo "ERROR: ibv_devinfo failed — IB drivers may not be loaded on this node"
        exit 1
    }
    active_ports=\$(blaunch -z \${hosts[0]} "ibstat 2>/dev/null | grep -c 'State: Active'" 2>/dev/null)
    # grep -c prints the count itself (also 0); anything else means ibstat failed
    [[ "\$active_ports" =~ ^[0-9]+\$ ]] || active_ports=0
    echo "Active IB ports on head node: \$active_ports (expected 8)"
    if [ "\$active_ports" -lt 8 ]; then
        echo "WARNING: fewer than 8 IB ports active — some rails may be down"
    fi
fi

port=\$(getfreeport)
head_node=\${hosts[0]}
echo ""
echo "Head node: \$head_node  Port: \$port"
echo "Per-job Ray ports: agent \$agent_grpc_port/\$agent_listen_port  dashboard \$dashboard_port  workers \$min_worker_port-\$max_worker_port"

# Path to NCCL env file — sourced by ray start shells so the daemon
# (and all workers it spawns) inherit the vars.
# (NCCL_ENV_FILE is set above; empty on Ethernet queues.)

echo ""
echo "================================================================"
echo "Starting Ray cluster..."
echo "================================================================"

blaunch -z \$head_node "
    if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
    export PYTHONUNBUFFERED=1
    if [ -n '\$NCCL_ENV_FILE' ] && [ -f '\$NCCL_ENV_FILE' ]; then source \$NCCL_ENV_FILE; fi
    \$GPU_ENV_FIX
    export LSF_RAY_JOB_TAG=\$RAY_JOB_TAG
    ${STALE_STOP}
    ray start --head --port \$port \
        \$client_flag \
        --num-cpus=${CPUS_PER_NODE} \
        --num-gpus=${GPUS_PER_NODE} \
        --temp-dir=\$RAY_TMPDIR \
        --dashboard-port=\$dashboard_port \
        --dashboard-agent-grpc-port=\$agent_grpc_port \
        --dashboard-agent-listen-port=\$agent_listen_port \
        --min-worker-port=\$min_worker_port \
        --max-worker-port=\$max_worker_port
" &

head_launch_pid=\$!
wait_start=\$SECONDS
sleep 20
waited=20
while ! timeout 30 ray status --address \$head_node:\$port 2>/dev/null; do
    # 'ray start' returns once the head is up; a non-zero exit means it failed
    if ! kill -0 \$head_launch_pid 2>/dev/null && ! wait \$head_launch_pid; then
        echo "ERROR: 'ray start --head' failed on \$head_node — see the error above"
        exit 1
    fi
    if [ \$waited -ge \$RAY_START_TIMEOUT ]; then
        echo "ERROR: Ray head node on \$head_node did not start within \${RAY_START_TIMEOUT}s"
        echo "       Check --venv (${VENV_PATH}) and the 'ray start' errors above."
        exit 1
    fi
    echo "Waiting for Ray head node..."
    sleep 3
    waited=\$((SECONDS - wait_start))
done
echo "Ray head node ready"

# Number of live nodes the head reports ("Active:" section of ray status).
# Prints 0 if the output cannot be parsed.
function ray_node_count() {
    timeout 30 ray status --address \$head_node:\$port 2>/dev/null | awk '
        /^(Active|Healthy):/ { in_active = 1; next }
        /^[A-Za-z]/          { in_active = 0 }
        in_active && \$1 ~ /^[0-9]+\$/ { n += \$1 }
        END { print n + 0 }'
}

if [ ${NUM_NODES} -gt 1 ]; then
    workers=("\${hosts[@]:1}")
    for host in "\${workers[@]}"; do
        blaunch -z \$host "
            if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
            export PYTHONUNBUFFERED=1
            if [ -n '\$NCCL_ENV_FILE' ] && [ -f '\$NCCL_ENV_FILE' ]; then source \$NCCL_ENV_FILE; fi
            \$GPU_ENV_FIX
            export LSF_RAY_JOB_TAG=\$RAY_JOB_TAG
            ${STALE_STOP}
            ray start --address \$head_node:\$port \
                --num-cpus=${CPUS_PER_NODE} \
                --num-gpus=${GPUS_PER_NODE} \
                --temp-dir=\$RAY_TMPDIR \
                --dashboard-agent-grpc-port=\$agent_grpc_port \
                --dashboard-agent-listen-port=\$agent_listen_port \
                --min-worker-port=\$min_worker_port \
                --max-worker-port=\$max_worker_port
        " &
        wait_start=\$SECONDS
        sleep 10
        waited=10
        while ! timeout 60 blaunch -z \$host ray status --address \$head_node:\$port 2>/dev/null; do
            if [ \$waited -ge \$RAY_START_TIMEOUT ]; then
                echo "ERROR: worker \$host could not reach the Ray head within \${RAY_START_TIMEOUT}s"
                exit 1
            fi
            echo "Waiting for worker \$host..."
            sleep 3
            waited=\$((SECONDS - wait_start))
        done
        echo "Worker \$host started"
    done

    # The check above only proves the head is reachable from each worker.
    # Confirm that every node actually joined before starting the script.
    wait_start=\$SECONDS
    waited=0
    nodes_up=\$(ray_node_count)
    while [ "\$nodes_up" -gt 0 ] && [ "\$nodes_up" -lt \${#hosts[@]} ]; do
        if [ \$waited -ge \$RAY_START_TIMEOUT ]; then
            echo "ERROR: only \$nodes_up of \${#hosts[@]} nodes joined the Ray cluster within \${RAY_START_TIMEOUT}s"
            ray status --address \$head_node:\$port
            exit 1
        fi
        echo "Waiting for nodes to join: \$nodes_up/\${#hosts[@]}"
        sleep 3
        waited=\$((SECONDS - wait_start))
        nodes_up=\$(ray_node_count)
    done
    if [ "\$nodes_up" -eq 0 ]; then
        echo "WARNING: could not read the node count from 'ray status' — not verified"
    else
        echo "All \$nodes_up nodes joined"
    fi
fi

echo ""
echo "================================================================"
echo "Ray cluster ready"
ray status --address \$head_node:\$port
echo "================================================================"
if [ "${NETWORK_BACKEND}" = "IB" ]; then
    echo "Tip: grep 'NET/IB' in the .out file to confirm NCCL selected InfiniBand"
    echo "     If you see NET/Socket, the HCA names may have changed — run ibv_devinfo"
fi
echo ""

RAY_ADDRESS=\$head_node:\$port python ${PYTHON_SCRIPT} ${SCRIPT_ARGS}
script_exit_code=\$?
echo "Script exited: \$script_exit_code"

cleanup

if [ \$script_exit_code -ne 0 ]; then
    echo "================================================================"
    echo "JOB FAILED — exit code: \$script_exit_code"
    echo "================================================================"
    exit \$script_exit_code
else
    echo "================================================================"
    echo "JOB COMPLETED SUCCESSFULLY"
    echo "================================================================"
fi
EOF
