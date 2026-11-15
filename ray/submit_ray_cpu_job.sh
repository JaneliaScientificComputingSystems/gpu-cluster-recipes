#!/bin/bash
#===============================================================================
# CPU-Only Ray Cluster Job Submission Script
# Janelia HPC — launches a Ray cluster on CPU-only LSF queues
#
# Usage:
#   ./submit_ray_cpu_job.sh --num-cpus=N --script=SCRIPT [options] [-- script_args...]
#
# Examples:
#   # One whole node on cpu_parallel (128 CPUs):
#   ./submit_ray_cpu_job.sh --num-cpus=128 --script=my_pipeline.py
#
#   # Two whole nodes (256 CPUs):
#   ./submit_ray_cpu_job.sh --num-cpus=256 --script=distributed_etl.py -- --input /data
#
#   # 8 CPUs — not a whole node, so use the local queue:
#   ./submit_ray_cpu_job.sh --queue=local --num-cpus=8 --script=my_pipeline.py
#
#   # Quick test on short queue (1 hour max):
#   ./submit_ray_cpu_job.sh --queue=short --num-cpus=4 --script=test.py
#
# Queue:
#   cpu_parallel -> whole nodes only (--num-cpus in multiples of 128),
#                   app profile parallel-128   [default]
#   any other queue (e.g. short) is submitted without an app profile
#===============================================================================

usage() {
    echo "Usage: $0 --num-cpus=N --script=SCRIPT [options] [-- script_args...]"
    echo ""
    echo "Required:"
    echo "  --num-cpus=N         Total CPUs to request"
    echo "  --script=FILE        Python script to run"
    echo ""
    echo "Optional:"
    echo "  --queue=QUEUE        LSF queue (default: cpu_parallel)"
    echo "  --job-name=NAME      Job name (default: ray_cpu_job)"
    echo "  --walltime=TIME      Walltime HH:MM (default: 24:00, 1:00 on short)"
    echo "  --venv=PATH          Python venv path"
    echo ""
    echo "Script arguments: use '--' separator"
    echo "  Example: $0 --num-cpus=256 --script=etl.py -- --input /data"
    exit 1
}

if [ $# -lt 1 ]; then usage; fi

QUEUE_NAME="cpu_parallel"
PYTHON_SCRIPT=""
JOB_NAME="ray_cpu_job"
WALLTIME=""
VENV_PATH=""
SCRIPT_ARGS=""
TOTAL_CPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue=*)    QUEUE_NAME="${1#*=}"; shift ;;
        --script=*)   PYTHON_SCRIPT="${1#*=}"; shift ;;
        --num-cpus=*) TOTAL_CPUS="${1#*=}"; shift ;;
        --job-name=*) JOB_NAME="${1#*=}"; shift ;;
        --walltime=*) WALLTIME="${1#*=}"; shift ;;
        --venv=*)     VENV_PATH="${1#*=}"; shift ;;
        --) shift; SCRIPT_ARGS=$(printf '%q ' "$@"); break ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$TOTAL_CPUS" ]; then echo "ERROR: --num-cpus is required"; usage; fi
if [ -z "$PYTHON_SCRIPT" ]; then echo "ERROR: --script is required"; usage; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "ERROR: Script not found: $PYTHON_SCRIPT"; exit 1; fi

# Site policy: cpu_parallel allocates whole nodes (128 slots) only
if [ "$QUEUE_NAME" = "cpu_parallel" ] && [ $((TOTAL_CPUS % 128)) -ne 0 ]; then
    echo "ERROR: cpu_parallel takes whole nodes only — --num-cpus must be a multiple of 128."
    echo "       For smaller jobs use --queue=local (or --queue=short for up to 1 hour)."
    exit 1
fi

# The short queue has a 1-hour limit
if [ -z "$WALLTIME" ]; then
    if [ "$QUEUE_NAME" = "short" ]; then WALLTIME="1:00"; else WALLTIME="24:00"; fi
fi

echo "================================================================"
echo "Ray CPU Cluster Job Submission"
echo "================================================================"
echo "Script:          $PYTHON_SCRIPT"
echo "Script args:     $SCRIPT_ARGS"
echo "Queue:           $QUEUE_NAME"
echo "Total CPUs:      $TOTAL_CPUS"
echo "Walltime:        $WALLTIME"
echo "================================================================"

# The parallel-128 app profile belongs to the cpu_parallel queue (see header)
APP_LINE=""
if [ "$QUEUE_NAME" = "cpu_parallel" ]; then
    APP_LINE="#BSUB -app parallel-128"
fi

# LSF does not create the log directory; without it the job output is lost
mkdir -p ../output

cat << EOF | bsub
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -n ${TOTAL_CPUS}
#BSUB -q ${QUEUE_NAME}
${APP_LINE}
#BSUB -o ../output/${JOB_NAME}_%J.out
#BSUB -e ../output/${JOB_NAME}_%J.err
#BSUB -W ${WALLTIME}

if [ -n "${VENV_PATH}" ]; then source ${VENV_PATH}/bin/activate; fi

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

# Discover hosts and CPUs per host from LSF allocation
declare -A host_cpus
for host in \$(cat \$LSB_DJOB_HOSTFILE); do
    host_cpus[\$host]=\$(( \${host_cpus[\$host]:-0} + 1 ))
done
hosts=(\$(echo "\${!host_cpus[@]}" | tr ' ' '\n' | sort))
num_nodes=\${#hosts[@]}

echo "Allocated nodes: \$num_nodes"
for host in "\${hosts[@]}"; do
    echo "  \$host: \${host_cpus[\$host]} CPUs"
done

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

port=\$(getfreeport)
head_node=\${hosts[0]}
echo ""
echo "Head node: \$head_node  Port: \$port"
echo "Per-job Ray ports: agent \$agent_grpc_port/\$agent_listen_port  dashboard \$dashboard_port"

echo ""
echo "================================================================"
echo "Starting Ray cluster..."
echo "================================================================"

blaunch -z \$head_node "
    if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
    export LSF_RAY_JOB_TAG=\$RAY_JOB_TAG
    ray start --head --port \$port \
        --num-cpus=\${host_cpus[\$head_node]} \
        --num-gpus=0 \
        --temp-dir=\$RAY_TMPDIR \
        --dashboard-port=\$dashboard_port \
        --dashboard-agent-grpc-port=\$agent_grpc_port \
        --dashboard-agent-listen-port=\$agent_listen_port
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

if [ \$num_nodes -gt 1 ]; then
    workers=("\${hosts[@]:1}")
    for host in "\${workers[@]}"; do
        blaunch -z \$host "
            if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
            export LSF_RAY_JOB_TAG=\$RAY_JOB_TAG
            ray start --address \$head_node:\$port \
                --num-cpus=\${host_cpus[\$host]} \
                --num-gpus=0 \
                --temp-dir=\$RAY_TMPDIR \
                --dashboard-agent-grpc-port=\$agent_grpc_port \
                --dashboard-agent-listen-port=\$agent_listen_port
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
    while [ "\$nodes_up" -gt 0 ] && [ "\$nodes_up" -lt \$num_nodes ]; do
        if [ \$waited -ge \$RAY_START_TIMEOUT ]; then
            echo "ERROR: only \$nodes_up of \$num_nodes nodes joined the Ray cluster within \${RAY_START_TIMEOUT}s"
            ray status --address \$head_node:\$port
            exit 1
        fi
        echo "Waiting for nodes to join: \$nodes_up/\$num_nodes"
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
