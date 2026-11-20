# torchrun — multi-node PyTorch launcher + NCCL/InfiniBand tools

Multi-node PyTorch via **`torchrun` + LSF `blaunch`**, for code that uses plain PyTorch DDP/FSDP.
`submit_torchrun_job.sh` parses the LSF host allocation, picks a master
address/port, launches `torchrun` on every node via `blaunch` (`rdzv_backend=c10d`), and
auto-configures NCCL per queue — InfiniBand queues (**B300 at XDR 800 Gb/s**, H100/H200 at NDR
400 Gb/s) source [`../nccl-ib/nccl_ib.sh`](../nccl-ib/) (or `nccl_ib_b300.sh` on B300) and run an IB
pre-flight check; Ethernet queues (L4/A100) fall back to socket NCCL.

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

### Running the workers in a sandbox (`--sandbox`)

`--sandbox=bwrap` or `--sandbox=podman` starts each node's worker inside a sandbox from
[agentic-sandbox](https://github.com/JaneliaScientificComputingSystems/agentic-sandbox), one per
node. The launcher itself stays outside. Use it for training code you did not write yourself,
such as code an AI agent wrote: the worker sees the working directory, the output directory and
what you name with `--ro`/`--rw`, and nothing else of your account (no home directory, keys or
tokens).

```bash
# Bubblewrap, with your venv
./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --script=train.py --venv=~/torch_env \
    --sandbox=bwrap --ro=/nrs/ml_datasets

# podman, with the image's own PyTorch (no --venv)
./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --script=train.py --sandbox=podman
```

| Option | Meaning |
|---|---|
| `--sandbox=bwrap\|podman` | which sandbox |
| `--sandbox-dir=PATH` | where `sandbox-run.sh` / `podman-run.sh` live; default `$AGENTIC_SANDBOX_DIR`, else `PATH` |
| `--image=IMAGE` | podman only; default `agentic-sandbox-gpu` (PyTorch, NCCL, Ray, InfiniBand libraries) |
| `--ro=PATH`, `--rw=PATH` | extra directories the worker may read / write (repeatable) |

The launcher picks the network from the queue; there is nothing to choose:

| Queue | Network inside the sandbox |
|---|---|
| InfiniBand (B300, H200, H100) | the other nodes of this job only: no internet, no other host. NCCL's data goes over InfiniBand directly, at full speed |
| Ethernet (L4, A100, RTX 6000) | the node's own network, not limited. NCCL's data travels over TCP there, and the limited network is about 40 times slower |
| one node | none |

Limits:

- `--conda` and `--modules` do not work with `--sandbox`. Use `--venv` with `bwrap`, or an image
  with `podman`. A host venv does not work inside a container.
- With `podman` the image is pulled on every node before the workers start (about two minutes
  for the default image on a node that does not have it yet).
- Measured on two H200 nodes with `test_torchrun_ibstress.py`: 481 GB/s sustained with either
  sandbox, against 482 without.

## Example: ViT on ImageNet-1K (`vit_imagenet_torchrun.py`)

Trains a Vision Transformer on ImageNet-1K with plain `torchrun` + DDP.

### What it does

Trains a Vision Transformer on ImageNet-1K with DDP + AdamW, bf16 mixed precision on H100/H200,
LR scaled by effective batch size with linear warmup + cosine decay, gradient clipping 1.0.
Sizes via `--model-size`: `base` (86M), `large` (307M), `huge` (632M).

### What it produces

- Per-epoch top-1 accuracy, loss, and images/sec throughput (to stderr)
- Checkpoints (`.pth`) saved to `../models/` with `--save-models`
- Checkpoints are plain PyTorch `state_dict`s — load them with any PyTorch inference script

### Data

ImageNet-1K, pre-installed at `/nrs/ml_datasets/imagenet` (shared NFS, ~138 GB) — no setup needed.

**Data-loading design:** each rank loads only its own ~1/N slice of the parquet shards into a
compact Arrow table and decodes rows lazily, so no process materializes the whole dataset. (An
earlier version eagerly `to_pylist()`-ed all ~1.2M images on *every* rank; rank 0 never reached
`DDP()` before the 600 s NCCL rendezvous timeout, and the job died with a c10d store timeout.) Train
shards are partitioned across ranks — standard DDP data parallelism — while validation keeps the
full shard set with a `DistributedSampler` for a correct all-reduced accuracy. Because ranks can hold
slightly different shard counts, each epoch is capped to the global-minimum batch count so the
gradient all-reduce stays in lock-step.

### Train

> ✅ Validated 2026-08-08 (2 nodes / 16×H100): dataset loads in ~19 s, reaches DDP init, trains at
> ~6–9k img/s over InfiniBand, and runs validation to a clean exit (both workers exit 0).



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
| `test_torchrun.py` | Minimal NCCL sanity check — all-reduce/all-gather across every rank; fast (~30 s once running). |
| `test_torchrun_soak.py` | Sustained small-message all-reduce over minutes — catches intermittent mid-run cross-node stalls the minimal test misses. Env tunables: `SOAK_ITERS`, `SOAK_NUMEL`, `SOAK_SLEEP`. |
| `test_torchrun_ibstress.py` | Bandwidth stress — message-size sweep + sustained large-message all-reduce, reports algbw/busbw (GB/s) like `nccl-tests`. Args: `--sweep-min-mb`, `--max-mb`, `--secs`, `--dtype`. |

### Sanity check

```bash
# Single node, 8 L4 GPUs (Ethernet)
./submit_torchrun_job.sh 1 --queue=gpu_l4_parallel --venv=~/torch_env --script=test_torchrun.py

# 2 nodes, 16 H200 GPUs (InfiniBand)
./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --venv=~/torch_env --script=test_torchrun.py
```
Success prints `[PASS] ... all ranks` and, on IB, `NET/IB/*/GDRDMA` in the NCCL debug output.

> ✅ Validated 2026-08-08: `test_torchrun.py` passes across all ranks (a 2-node/16-GPU run measured
> ~574 GB/s on the sanity all-reduce).

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

This repo used to ship `test_gloo_barrier.py`, a reproduction of the abort and of the fix. It was
removed on 2026-11-19: it held two whole nodes idle for minutes per run and depended on watchdog
behaviour that changes between PyTorch versions. The last version is in the git history.

### Bandwidth stress (busbw)

```bash
# 4 nodes / 32 GPUs on B300 — sweep to 4 GB + 10-min sustained hammer
./submit_torchrun_job.sh 4 --queue=gpu_b300_parallel --venv=~/torch_env \
    --job-name=ibstress --walltime=0:30 \
    --script=test_torchrun_ibstress.py -- --sweep-min-mb 8 --max-mb 4096 --secs 600
```

**Validated (2026-08-08, 32×B300):** busbw plateaus **~740 GB/s** (≈92% of the 8×XDR 800 Gb/s
per-node line rate; ~96 Tb/s aggregate across 128 rails) and holds flat over a 10-min sustained
hammer with zero errors. The soak test likewise ran 2000 cross-node all-reduces with no hang.
