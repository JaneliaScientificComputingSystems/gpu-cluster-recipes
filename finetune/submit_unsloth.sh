#!/bin/bash
#===============================================================================
# Unsloth single-GPU LoRA fine-tune — LSF submit wrapper
# Janelia HPC
#
# Unsloth fine-tuning is SINGLE-GPU (one process, one GPU). This wrapper submits
# unsloth_gemma4_lora.py to one GPU on a non-parallel queue and points all the
# HuggingFace / cache / temp dirs at node-local /scratch (the shared /misc and
# /nrs trees are read-only for caching).
#
# Usage:
#   ./submit_unsloth.sh [options] [-- extra_script_args...]
#
# Examples:
#   # Smoke test (few steps, tiny slice) on an H100:
#   ./submit_unsloth.sh --max-samples=100 --max-steps=5
#
#   # Real adapter from your own data on an H200:
#   ./submit_unsloth.sh --queue=gpu_h200 --dataset=/path/to/mydata.jsonl \
#       --output=../models/gemma4_myadapter -- --epochs=3 --max-steps=-1
#
# Queue guidance: Gemma-4-26B in 16-bit needs ~52 GB resident + optimizer/activations,
# so use an 80 GB-class GPU: gpu_h100 (recommended), gpu_h200, or gpu_b300.
#===============================================================================
set -euo pipefail

QUEUE="gpu_h100"
VENV="$HOME/unsloth_env"
SCRIPT="unsloth_gemma4_lora.py"
JOB_NAME="unsloth_gemma4"
WALLTIME="24:00"
CPUS=12                       # queue slot ratio is ~12 CPUs per GPU
OUTPUT="../models/gemma4_lora"
PASS_ARGS=()
SCRIPT_OPTS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --queue=*)       QUEUE="${1#*=}" ;;
        --venv=*)        VENV="${1#*=}" ;;
        --script=*)      SCRIPT="${1#*=}" ;;
        --job-name=*)    JOB_NAME="${1#*=}" ;;
        --walltime=*)    WALLTIME="${1#*=}" ;;
        --num-cpus=*)    CPUS="${1#*=}" ;;
        --output=*)      OUTPUT="${1#*=}"; SCRIPT_OPTS+=("--output" "${1#*=}") ;;
        --model=*)       SCRIPT_OPTS+=("--model" "${1#*=}") ;;
        --dataset=*)     SCRIPT_OPTS+=("--dataset" "${1#*=}") ;;
        --max-samples=*) SCRIPT_OPTS+=("--max-samples" "${1#*=}") ;;
        --max-steps=*)   SCRIPT_OPTS+=("--max-steps" "${1#*=}") ;;
        --epochs=*)      SCRIPT_OPTS+=("--epochs" "${1#*=}") ;;
        --)              shift; PASS_ARGS=("$@"); break ;;
        -h|--help)       sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

case "$QUEUE" in
    gpu_h100|gpu_h200|gpu_b300) ;;
    *) echo "WARNING: '$QUEUE' is not an 80 GB-class GPU queue — Gemma-4-26B 16-bit may not fit." >&2 ;;
esac

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "$(cd "$(dirname "$0")/.." && pwd)")"
OUTDIR="$REPO_ROOT/output"
mkdir -p "$OUTDIR"

echo "Queue:     $QUEUE  (1 GPU, $CPUS CPUs)"
echo "Env:       $VENV"
echo "Script:    $SCRIPT  ${SCRIPT_OPTS[*]:-} ${PASS_ARGS[*]:-}"
echo "Adapters:  $OUTPUT"

bsub -q "$QUEUE" -J "$JOB_NAME" -n "$CPUS" \
     -gpu "num=1:mode=exclusive_process" -W "$WALLTIME" \
     -o "$OUTDIR/${JOB_NAME}_%J.out" -e "$OUTDIR/${JOB_NAME}_%J.err" \
     bash -c "
        set -e
        # All caches/temp on node-local /scratch (shared trees are read-only for writes)
        export HF_HOME=/scratch/\$(whoami)/hf_home
        export TMPDIR=/scratch/\$(whoami)/tmp
        export XDG_CACHE_HOME=/scratch/\$(whoami)/xdg_cache
        mkdir -p \"\$HF_HOME\" \"\$TMPDIR\" \"\$XDG_CACHE_HOME\"
        source '$VENV/bin/activate'
        cd '$(cd "$(dirname "$0")" && pwd)'
        echo \"HOST \$(hostname -s); GPU:\"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        python '$SCRIPT' ${SCRIPT_OPTS[*]:-} ${PASS_ARGS[*]:-}
     "
