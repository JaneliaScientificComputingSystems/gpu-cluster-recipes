# torchrun — multi-node PyTorch launcher + NCCL/InfiniBand tools

Multi-node PyTorch via **`torchrun` + LSF `blaunch`**, for code that uses plain PyTorch DDP/FSDP.
`submit_torchrun_job.sh` parses the LSF host allocation, picks a master
address/port, launches `torchrun` on every node via `blaunch` (`rdzv_backend=c10d`), and
auto-configures NCCL per queue — InfiniBand queues (**B300 at XDR 800 Gb/s**, H100/H200 at NDR
400 Gb/s) source [`../nccl-ib/nccl_ib.sh`](../nccl-ib/) (or `nccl_ib_b300.sh` on B300) and run an IB
pre-flight check; Ethernet queues (L4/A100/RTX 6000) fall back to socket NCCL.

## Environment

Build a PyTorch venv for these examples:

```bash
python -m venv ~/torch_env
source ~/torch_env/bin/activate
pip install torch torchvision datasets      # NCCL/IB tests need only torch
```

Pass it with `--venv=~/torch_env` (or point `--conda`/`--venv` at any environment that has PyTorch).

## The launcher — `submit_torchrun_job.sh`

```
./submit_torchrun_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]

Options:
  --queue=QUEUE        GPU queue (default: gpu_h100_parallel)
  --num-gpus=N         GPUs per node (non-parallel queues, default: 1)
  --num-cpus=N         CPUs to request (non-parallel queues, default: 12 per GPU)
  --venv=PATH          Python venv to activate
  --conda=NAME         Conda/mamba environment to activate
  --modules=M1,M2      Modules to load (e.g. cuda/12.8,gcc/12.3)
  --workdir=PATH       Working directory (default: script's directory)
  --job-name=NAME      Job name (default: torchrun_job)
  --walltime=TIME      Walltime H:MM (default: 24:00 non-parallel, auto for parallel)
  --output-dir=PATH    Output log directory (default: ../output)
  --host="H1 H2 ..."   Pin the job to specific hosts (#BSUB -m)

Script args go after '--':
  ./submit_torchrun_job.sh 2 --script=train.py -- --epochs=50 --batch-size=128
```

It also runs training code from other projects — point `--conda`/`--workdir` at your own
environment and code:

```bash
./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel \
    --conda=RNAnix --modules=cuda/12.8,gcc/12.3 \
    --workdir=~/RNAnix --script=~/RNAnix/runner/train_rna.py \
    -- --dtype bf16 --max_steps 50000
```

---

**In a sandbox:** add `--sandbox=bwrap` or `--sandbox=podman` to run code you did not write
yourself (such as code an AI agent wrote) where it cannot see your home directory, keys or
tokens. See [Running the workers in a sandbox](#running-the-workers-in-a-sandbox---sandbox).

## Example: ViT on ImageNet-1K (`vit_imagenet_torchrun.py`)

Trains a Vision Transformer on ImageNet-1K with DDP + AdamW, bf16 mixed precision on H100/H200,
LR scaled by effective batch size with linear warmup + cosine decay, gradient clipping 1.0.
Sizes via `--model-size`: `base` (86M), `large` (307M), `huge` (632M).

With `--save-models`, checkpoints go to `../models/` as plain PyTorch `state_dict`s.

Data: `/nrs/ml_datasets/imagenet` (`--data-dir` to change). Each rank loads only its own slice of
the parquet shards and decodes rows lazily, so no process holds the whole dataset.

### Train

```bash
# ViT-base, 2 nodes / 16 GPUs — validation run (run from this torchrun/ directory)
./submit_torchrun_job.sh 2 --queue=gpu_h100_parallel --venv=~/torch_env \
    --job-name=vit_torchrun \
    --script=vit_imagenet_torchrun.py -- \
    --model-size=base --epochs=5 --batch-size=64 --save-models

# ViT-large, single node / 4 GPUs — development
./submit_torchrun_job.sh 1 --queue=gpu_h100 --num-gpus=4 --venv=~/torch_env \
    --script=vit_imagenet_torchrun.py -- \
    --model-size=large --epochs=90 --save-models
```

---

## NCCL / InfiniBand tests

Diagnostics for the interconnect, all launched via `submit_torchrun_job.sh` (run from this
directory, so `--script=<file>` is relative):

| Script | Purpose |
|---|---|
| `test_torchrun.py` | Minimal NCCL sanity check — all-reduce/all-gather across every rank; fast. |
| `test_torchrun_soak.py` | Sustained small-message all-reduce over minutes — catches intermittent mid-run cross-node stalls the minimal test misses. Env tunables: `SOAK_ITERS`, `SOAK_NUMEL`, `SOAK_SLEEP`. |
| `test_torchrun_ibstress.py` | Bandwidth stress — message-size sweep + sustained large-message all-reduce, reports algbw/busbw (GB/s) like `nccl-tests`. Args: `--sweep-min-mb`, `--max-mb`, `--secs`, `--dtype`. |

### Sanity check

```bash
# Single node, 8 L4 GPUs (Ethernet). One node goes to the non-parallel queue: the
# *_parallel queues are for jobs of two or more nodes, and the launcher refuses one node there
./submit_torchrun_job.sh 1 --queue=gpu_l4 --num-gpus=8 --venv=~/torch_env --script=test_torchrun.py

# 2 nodes, 16 H200 GPUs (InfiniBand)
./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --venv=~/torch_env --script=test_torchrun.py
```
Success prints `[PASS]` lines for all-reduce, all-gather and bandwidth and, on IB, `NET/IB/*/GDRDMA` in the NCCL debug output.

### Pitfall: a long solo phase on rank 0

If rank 0 works alone for a long time while the other ranks wait for it (a held-out evaluation, a
large checkpoint save), what the other ranks wait **on** decides whether the job survives.

- **Symptom.** A healthy multi-node job aborts while rank 0 is busy. The waiting ranks sit in an
  NCCL call such as `dist.barrier()`, and PyTorch's NCCL watchdog
  (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`) declares them stuck once the wait exceeds its timeout.
- **Timeout.** PyTorch's default is about 480 s, which is what the Ethernet queues get. On the
  InfiniBand queues `../nccl-ib/` sets it to 3600 s. A value you export before submitting wins.
- **Fix.** Wait on a Gloo group, which the NCCL watchdog does not cover, and use NCCL again
  afterwards:

```python
from datetime import timedelta
import torch.distributed as dist

gloo_group = dist.new_group(backend="gloo")      # once, after init_process_group
...
dist.monitored_barrier(group=gloo_group, timeout=timedelta(seconds=7200), wait_all_ranks=True)
```

- **Or** raise the watchdog above the longest solo phase: `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`
  before submitting.

### Bandwidth stress (busbw)

```bash
# 4 nodes / 32 GPUs on B300 — sweep to 4 GB + 10-min sustained hammer
./submit_torchrun_job.sh 4 --queue=gpu_b300_parallel --venv=~/torch_env \
    --job-name=ibstress --walltime=0:30 \
    --script=test_torchrun_ibstress.py -- --sweep-min-mb 8 --max-mb 4096 --secs 600
```

---

## Running the workers in a sandbox (`--sandbox`)

`--sandbox=bwrap` or `--sandbox=podman` starts each node's worker inside a sandbox from
[agentic-sandbox](https://github.com/JaneliaScientificComputingSystems/agentic-sandbox), one per
node. The launcher itself stays outside. Use it for training code you did not write yourself,
such as code an AI agent wrote. Everything else in this README works the same without it.

### Before you start

1. Clone agentic-sandbox (25 November 2026 or later; `git pull` an older checkout) and point the
   launcher at its `scripts/` directory:

   ```bash
   git clone https://github.com/JaneliaScientificComputingSystems/agentic-sandbox.git ~/agentic-sandbox
   export AGENTIC_SANDBOX_DIR=~/agentic-sandbox/scripts      # or pass --sandbox-dir=...
   ```

2. `--sandbox=bwrap` needs nothing more. It uses your own venv (`--venv`).
3. `--sandbox=podman` needs podman set up for your account once: follow section 3 of
   agentic-sandbox's
   [SETUP.md](https://github.com/JaneliaScientificComputingSystems/agentic-sandbox/blob/main/SETUP.md).
   It uses the image's Python, not a venv. The default image has PyTorch and torchvision.

### Run it

Take a command from this README and add `--sandbox=...`, plus `--ro=` for a dataset outside
the working directory:

```bash
# Bubblewrap, with your venv
./submit_torchrun_job.sh 2 --queue=gpu_h100_parallel --venv=~/torch_env \
    --sandbox=bwrap --ro=/nrs/ml_datasets \
    --script=vit_imagenet_torchrun.py -- \
    --model-size=base --epochs=5 --batch-size=64 --save-models

# podman, with the image's own PyTorch (no --venv)
./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --sandbox=podman --script=test_torchrun.py
```

| Option | Meaning |
|---|---|
| `--sandbox=bwrap\|podman` | which sandbox |
| `--sandbox-dir=PATH` | where `sandbox-run.sh` / `podman-run.sh` live; default `$AGENTIC_SANDBOX_DIR`, else `PATH` |
| `--image=IMAGE` | podman only; default `agentic-sandbox-gpu` |
| `--ro=PATH`, `--rw=PATH` | extra directories the worker may read / write (repeatable) |

### What the sandbox sees

| Directory | Access | Who adds it |
|---|---|---|
| the working directory (`--workdir`, default the script's directory) | read and write | the launcher |
| the output directory (job logs) | read and write | the launcher |
| `../models` beside the working directory (checkpoints from `--save-models`) | read and write | the launcher: always for code in this repository, otherwise only if the directory exists |
| the script's directory and, with `bwrap`, your venv and the Python it was made from (uv, pyenv) | read | the launcher |
| datasets such as `/nrs/ml_datasets`, and anything else the script reads or writes | | you: `--ro=` / `--rw=` |

Nothing else of your account is visible: no home directory, keys or tokens.

### The network

The launcher picks the network; there is nothing to choose:

| Job | Network inside the sandbox |
|---|---|
| 2+ nodes | the node's own network, not limited, so that the nodes reach each other. Files, home directory, keys and tokens stay hidden as always |
| 2+ nodes on an InfiniBand queue (B300, H200, H100) | the same, with the InfiniBand devices: NCCL's data goes over InfiniBand directly |
| one node | none (on an InfiniBand queue the InfiniBand devices are still there) |

### Limits

- `--conda` and `--modules` do not work with `--sandbox`. Use `--venv` with `bwrap`, or an image
  with `podman`. A host venv does not work inside a container.
- With `podman` the image is pulled on every node before the workers start.
- The cluster accepts a multi-node GPU job only on a `*_parallel` queue and only with whole
  nodes, which is what this launcher asks for. Each sandbox therefore has its node to itself.
