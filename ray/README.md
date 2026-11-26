# Ray Distributed Training Examples

Distributed PyTorch training and data processing with Ray on Janelia's HPC cluster. GPU examples use InfiniBand with GPUDirect RDMA for high-performance multi-node training — **XDR (800 Gb/s) on B300**, NDR (400 Gb/s) on H200/H100. CPU examples use Ray Data pipelines for distributed ETL and analysis.

Six ready-to-run examples:

| Example | Model | Dataset | Purpose |
|---------|-------|---------|---------|
| CIFAR-10 | ResNet-18 (11M) | 60K images, 170 MB | Quick smoke test |
| GPT-2 | GPT-2 small–2.7B | OpenWebText, 9 GB | Language model, DDP or FSDP |
| ImageNet ResNet | ResNet-50 (25.6M) | ImageNet-1K, 138 GB | Industry-standard CNN benchmark |
| ImageNet ViT | ViT base–huge (86M–632M) | ImageNet-1K, 138 GB | Vision Transformer |
| Cell Segmentation | Swin-B (88M) | LIVECell, 2 GB | Fine-tuning a pretrained model for microscopy — transfer learning |
| Sequence Search | — (CPU only) | UniRef90, 28 GB | Distributed protein sequence search — k-mer similarity against BLAST DB |

---

## GPU Queues

**Parallel queues** — whole nodes, multi-node training:

| Queue | GPU | GPUs/node | Network | CPUs/node |
|-------|-----|-----------|---------|-----------|
| `gpu_b300_parallel` | B300 | 8 | InfiniBand XDR 800 Gb/s | 96 |
| `gpu_h200_parallel` | H200 | 8 | InfiniBand NDR 400 Gb/s | 96 |
| `gpu_h100_parallel` | H100 | 8 | InfiniBand NDR 400 Gb/s | 96 |
| `gpu_rtx6000_parallel` | RTX PRO 6000 | 8 | Ethernet | 128 |
| `gpu_l4_parallel` | L4 | 8 | Ethernet | 64 |
| `gpu_a100_parallel` | A100 | 4 | Ethernet | 48 |

Parallel queues are for jobs of 2 or more whole nodes. Site policy: never submit a single-node or partial-node job to a `*_parallel` queue — `submit_ray_job.sh` and `run_inference.sh` refuse it.

**Non-parallel queues** — single node, request 1-8 GPUs with `--num-gpus`:

| Queue | GPU | Max GPUs | Network |
|-------|-----|----------|---------|
| `gpu_b300` | B300 | 8 | InfiniBand XDR 800 Gb/s |
| `gpu_h100` | H100 | 8 | InfiniBand NDR 400 Gb/s |
| `gpu_h200` | H200 | 8 | InfiniBand NDR 400 Gb/s |
| `gpu_l4` | L4 | 8 | Ethernet |
| `gpu_l4_16` | L4 | 8 | Ethernet |
| `gpu_rtx6000` | RTX PRO 6000 | 8 | Ethernet |
| `gpu_a100` | A100 | 4 | Ethernet |

**CPU queue** — multi-node, no GPUs:

| Queue | Network | CPUs/node |
|-------|---------|-----------|
| `cpu_parallel` | Ethernet | 128 |

`cpu_parallel` allocates whole nodes only: request CPUs in multiples of 128. For smaller jobs use `--queue=local` (or `--queue=short`, 1 hour max).

---

## Installation

```bash
git clone https://github.com/JaneliaScientificComputingSystems/gpu-cluster-recipes.git
cd gpu-cluster-recipes/ray      # run all commands below from this directory
chmod +x *.sh

python -m venv ~/ray_env
source ~/ray_env/bin/activate
pip install "ray[train,data]" torch torchvision

# For GPT-2 text generation (tiktoken) and prepare_openwebtext.sh (datasets)
pip install datasets tiktoken

# For the cell segmentation example
pip install scipy scikit-image pycocotools
```

---

## Submission Script

```
./submit_ray_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]

Options:
  --queue=QUEUE        GPU queue (default: gpu_h100_parallel)
  --num-gpus=N         GPUs per node (non-parallel queues, default: 1)
  --num-cpus=N         CPUs to request (non-parallel queues, default: the queue's slots per GPU —
                       l4=8, l4_16/rtx6000=16, l4_large=64, a100/h100/h200/b300=12)
  --venv=PATH          Python venv path
  --job-name=NAME      Job name (default: ray_job)
  --walltime=TIME      Walltime (default: 24h non-parallel, 4-8h parallel)

Script args go after '--':
  ./submit_ray_job.sh 2 --script=train.py -- --epochs=50
```

**Multi-node** (parallel queues — whole nodes, all GPUs, 2 or more nodes):
```bash
./submit_ray_job.sh 2 --queue=gpu_h100_parallel --venv=~/ray_env \
    --script=train.py -- --num-gpus=16 --num-nodes=2
```

**Single-node development** (non-parallel queues — request specific GPUs):
```bash
./submit_ray_job.sh 1 --queue=gpu_h100 --num-gpus=2 --venv=~/ray_env \
    --script=train.py -- --num-gpus=2 --num-nodes=1
```

**In a sandbox:** add `--sandbox=bwrap` or `--sandbox=podman` to run code you did not write
yourself (such as code an AI agent wrote) where it cannot see your home directory, keys or
tokens. See [Running Ray in a sandbox](#running-ray-in-a-sandbox---sandbox).

---

## CPU Submission Script

For CPU-only workloads (data pipelines, ETL, analysis) — no GPUs:

```
./submit_ray_cpu_job.sh --num-cpus=N --script=SCRIPT [options] [-- script_args...]

Options:
  --num-cpus=N         Total CPUs to request (required; multiple of 128 on cpu_parallel)
  --script=FILE        Python script to run (required)
  --queue=QUEUE        LSF queue (default: cpu_parallel)
  --venv=PATH          Python venv path
  --job-name=NAME      Job name (default: ray_cpu_job)
  --walltime=TIME      Walltime HH:MM (default: 24:00, 1:00 on short)

Script args go after '--':
  ./submit_ray_cpu_job.sh --num-cpus=256 --script=pipeline.py -- --input /data
```

On `cpu_parallel` each 128 CPUs is one whole node. The script discovers the allocation at runtime and starts Ray accordingly.

See [Example 6](#example-6-protein-sequence-search-cpu) for commands.

---

## Example 1: CIFAR-10

ResNet-18 (11M parameters) on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), trained with DDP.
The smallest example; use it to check that multi-node training works.

Data: `/nrs/ml_datasets/cifar10`. The training script has no `--data-dir`; the path is set in its
`main()`. `image_classifier.py` takes `--data-dir`.

### Train

```bash
# Short run — 10 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=10 --batch-size=256 --save-models

# Full run — 50 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=256 --save-models
```

### Inference

Give the classifier an image — it returns the top-3 predicted CIFAR-10 categories with confidence scores:

```bash
# Test on CIFAR-10 validation samples
./run_inference.sh --venv=~/ray_env --script=image_classifier.py -- \
    --model ../models/cifar10_resnet18_best.pth --test

# Classify your own image
./run_inference.sh --venv=~/ray_env --script=image_classifier.py -- \
    --model ../models/cifar10_resnet18_best.pth --image photo.jpg
```

---

## Example 2: GPT-2

GPT-2 on [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), trained with DDP or
FSDP, with gradient accumulation and mixed precision.

Data: tokenized OpenWebText at `/nrs/ml_datasets/openwebtext` (`--data-dir` to change;
`prepare_openwebtext.sh` builds a copy).

### Model sizes

Five model sizes are available via `--model-size`:

| Size | Layers | Heads | Dim | Params | Recommended mode |
|------|--------|-------|-----|--------|------------------|
| `small` | 12 | 12 | 768 | 117M | DDP |
| `medium` | 24 | 16 | 1024 | 345M | DDP |
| `large` | 36 | 20 | 1280 | 774M | DDP |
| `xl` | 48 | 25 | 1600 | 1.5B | DDP or FSDP |
| `2b` | 56 | 32 | 2048 | 2.7B | FSDP |

### Train

GPT-2 uses iteration count, not epochs. Each iteration processes `batch_size × grad_accum × num_gpus` sequences (1024 tokens each). With the defaults below, effective batch = 1,048,576 tokens/iter.

```bash
# Short run — GPT-2 small, 2000 iters
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=2000 --batch-size=8 --grad-accum=8 --save-models

# Full run — 50K iters (set --walltime: the default for 2 nodes is 4h)
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --walltime=8:00 \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=50000 --batch-size=8 --grad-accum=8 --save-models
```

Larger sizes take `--model-size` and, for `2b`, `--mode=fsdp`:

```bash
# GPT-2 2.7B — FSDP, 8 nodes / 64 GPUs
./submit_ray_job.sh 8 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=gpt2_2b \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --model-size=2b --mode=fsdp \
    --max-iters=500 --batch-size=2 --grad-accum=8 --save-models
```

#### Batch size reference by model size (H200, 141 GB)

| Size | DDP per-GPU batch | FSDP per-GPU batch |
|------|-------------------|--------------------|
| small (117M) | 8 | 8 |
| medium (345M) | 8 | 8 |
| large (774M) | 4–8 | 8 |
| xl (1.5B) | 2–4 | 4–8 |
| 2b (2.7B) | — | 2–4 |

### Evaluate & Generate

**Perplexity evaluation** — measures how well the model predicts held-out text (lower = better):

```bash
./run_inference.sh --venv=~/ray_env --script=gpt2_eval.py -- \
    --model ../models/gpt2_small_ddp_best.pth --num-batches 200

# For larger models, specify --model-size to match the checkpoint
./run_inference.sh --venv=~/ray_env --script=gpt2_eval.py -- \
    --model ../models/gpt2_2b_fsdp_best.pth --model-size=2b --num-batches 200
```

**Text generation** — GPT-2 is a text completion model, not a chatbot. Give it the start of a sentence and it continues writing in the style of web articles:

```bash
./run_inference.sh --venv=~/ray_env --script=gpt2_generate.py -- \
    --model ../models/gpt2_small_ddp_best.pth --prompt "The brain"
```

**Interactive mode** — start a GPU shell and run the generate script directly:

```bash
bsub -n8 -q gpu_l4 -gpu "num=1" -W 4:00 -Is /bin/bash
# once on the GPU node:
cd ~/Projects/gpu-cluster-recipes/ray
source ~/ray_env/bin/activate
python gpt2_generate.py --model ../models/gpt2_small_ddp_best.pth --interactive
```

---

## Example 3: ResNet-50 on ImageNet-1K

ResNet-50 (25.6M parameters) on [ImageNet-1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
trained with DDP and large-batch SGD in mixed precision. Data is streamed with Ray Data from
parquet shards.

Data: `/nrs/ml_datasets/imagenet` (`--data-dir` to change; `prepare_imagenet.sh` downloads a copy
and needs a Hugging Face token with the ImageNet license accepted).

### Train

```bash
# Short run — 3 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=resnet50 \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=3 --batch-size=128 --save-models

# Full run — 90 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=resnet50 \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=90 --batch-size=128 --save-models

# 8 nodes, 64 GPUs
./submit_ray_job.sh 8 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=resnet50 \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --epochs=90 --batch-size=64 --save-models
```

### Inference

Give the classifier an image — it returns the top-5 predicted ImageNet categories (out of 1,000) with confidence scores:

```bash
# Test on ImageNet validation samples
./run_inference.sh --venv=~/ray_env --script=imagenet_classifier.py -- \
    --model ../models/resnet50_imagenet_best.pth --test

# Classify your own image
./run_inference.sh --venv=~/ray_env --script=imagenet_classifier.py -- \
    --model ../models/resnet50_imagenet_best.pth --image photo.jpg
```

### Batch size reference

LR is automatically scaled: `lr = base_lr * effective_batch / 256`.

| GPU | Per-GPU batch |
|-----|--------------|
| H200 (141 GB) | 128–256 |
| H100 (80 GB) | 128 |

---

## Example 4: ViT on ImageNet-1K

Vision Transformer on ImageNet-1K, trained with DDP and AdamW. It uses the same data and data
pipeline as Example 3. Three sizes via `--model-size`:

| Size | Layers | Heads | Dim | Params |
|------|--------|-------|-----|--------|
| `base` | 12 | 12 | 768 | 86M |
| `large` | 24 | 16 | 1024 | 307M |
| `huge` | 32 | 16 | 1280 | 632M |

### Train

```bash
# ViT-Large, 2 nodes, 16 GPUs — short run
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=vit_large \
    --script=vit_imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --model-size=large --epochs=3 --batch-size=32 --save-models

# ViT-Large, full 90 epochs (the default walltime for 2 nodes is 4h — add --walltime=HH:MM sized to your run)
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=vit_large \
    --script=vit_imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --model-size=large --epochs=90 --batch-size=32 --save-models

# ViT-Huge, 7 nodes, 56 GPUs
./submit_ray_job.sh 7 --queue=gpu_h100_parallel --venv=~/ray_env \
    --job-name=vit_huge \
    --script=vit_imagenet_distributed_training.py -- \
    --num-gpus=56 --num-nodes=7 --model-size=huge --epochs=90 --batch-size=16 --save-models
```

### Inference

```bash
# Test on ImageNet validation samples
./run_inference.sh --venv=~/ray_env --script=vit_imagenet_classifier.py -- \
    --model ../models/vit_large_imagenet_best.pth --test

# Classify your own image
./run_inference.sh --venv=~/ray_env --script=vit_imagenet_classifier.py -- \
    --model ../models/vit_large_imagenet_best.pth --image photo.jpg
```

### Batch size reference (ViT-Large)

| GPU | Per-GPU batch |
|-----|--------------|
| H200 (141 GB) | 64–128 |
| H100 (80 GB) | 32–64 |
| A100 (80 GB) | 16–32 |

---

## Example 5: Cell Segmentation (Swin + LIVECell)

Transfer learning: a Swin Transformer pretrained on ImageNet is fine-tuned for cell segmentation
on [LIVECell](https://www.nature.com/articles/s41592-021-01249-6) microscopy images. The model
predicts background, cell interior and cell boundary per pixel; a watershed step turns that into
individual cells. To use it on your own microscopy data, replace the dataset.

Data: `/nrs/ml_datasets/livecell`, with precomputed masks in `masks/` (`--data-dir` to change;
`prepare_livecell.sh` and then `prepare_livecell_masks.py` build a copy).

### Train

```bash
# Fine-tune on 2 H100 nodes (16 GPUs) — 50 epochs
./submit_ray_job.sh 2 --queue=gpu_h100_parallel --venv=~/ray_env \
    --job-name=livecell \
    --script=livecell_finetune.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=4 --save-models

# Single A100 node (4 GPUs) — good for development
./submit_ray_job.sh 1 --queue=gpu_a100 --num-gpus=4 --venv=~/ray_env \
    --job-name=livecell \
    --script=livecell_finetune.py -- \
    --num-gpus=4 --num-nodes=1 --epochs=50 --batch-size=4 --save-models

# Train from scratch (no pretrained backbone) — for comparison
./submit_ray_job.sh 2 --queue=gpu_h100_parallel --venv=~/ray_env \
    --job-name=livecell_scratch \
    --script=livecell_finetune.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=4 --save-models --no-pretrained
```

### Inference

The segmenter takes a microscopy image and produces a colored overlay with individual cell instances:

```bash
# Test on LIVECell test images
./run_inference.sh --venv=~/ray_env --script=cell_segmenter.py -- \
    --model ../models/swin_livecell_best.pth --test

# Segment your own microscopy image
./run_inference.sh --venv=~/ray_env --script=cell_segmenter.py -- \
    --model ../models/swin_livecell_best.pth --image cells.tif

# Process a directory of images
./run_inference.sh --venv=~/ray_env --script=cell_segmenter.py -- \
    --model ../models/swin_livecell_best.pth --image-dir ./my_images/
```

Output segmented images are saved to `../output/segmentation/`. `run_inference.sh` runs the job in the `ray/` directory, so relative paths (`--model`, `--image`, `--image-dir`) are resolved against `ray/`.

---

## Example 6: Protein Sequence Search (CPU)

Searches query protein sequences against UniRef90 by k-mer similarity, a distributed, approximate
alternative to BLAST. The database is split into chunks, each CPU scores its chunk against the
queries (Jaccard similarity on shared k-mers), and the top hits per query are reported. No GPUs;
it runs on `cpu_parallel` with `submit_ray_cpu_job.sh`.

Data: `/misc/blast/db/uniref90/uniref90.fasta` (`--db` to change). `test_queries.fasta` here is a
sample query file.

### Run

```bash
# Test mode — picks 5 sample queries from the DB itself
./submit_ray_cpu_job.sh --num-cpus=128 --venv=~/ray_env \
    --script=sequence_search.py -- --test

# Search your own sequences
./submit_ray_cpu_job.sh --num-cpus=128 --venv=~/ray_env \
    --script=sequence_search.py -- --query my_proteins.fasta

# Multi-node — 256 CPUs (2 nodes) for faster search
./submit_ray_cpu_job.sh --num-cpus=256 --venv=~/ray_env \
    --job-name=seq_search \
    --script=sequence_search.py -- --query my_proteins.fasta --top-n=20
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--query` | — | Query FASTA file (required unless `--test`) |
| `--test` | — | Pick sample queries from the database |
| `--db` | `/misc/blast/db/uniref90/uniref90.fasta` | Database FASTA file |
| `--kmer-size` | 3 | K-mer size for similarity (3 = 8,000 possible protein 3-mers) |
| `--top-n` | 10 | Report top N hits per query |
| `--num-queries` | 5 | Number of sample queries in `--test` mode |
| `--chunk-mb` | 100 | Database chunk size in MB |

---

## Resume from checkpoint

All scripts support `--resume`:

```bash
# Resume CIFAR-10
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --save-models \
    --resume=../models/cifar10_resnet18_latest.pth

# Resume ImageNet
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=90 --batch-size=128 --save-models \
    --resume=../models/resnet50_imagenet_latest.pth

# Resume GPT-2
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --model-size=small --mode=ddp \
    --max-iters=5000 --save-models \
    --resume=../models/gpt2_small_ddp_latest.pth
```

Checkpoints: `--save-models` saves `<model>_latest.pth` (every epoch) and `<model>_best.pth` (on improvement) to `../models/`.

---

## Running Ray in a sandbox (`--sandbox`)

`--sandbox=bwrap` or `--sandbox=podman` runs Ray and your script inside a sandbox from
[agentic-sandbox](https://github.com/JaneliaScientificComputingSystems/agentic-sandbox), one per
node. The launcher and the job script stay outside. Use it for code you did not write yourself,
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
   It uses the image's Python, not a venv. The default image has PyTorch, Ray and the packages
   the examples here import.

### Run it

Take any example command in this README and add `--sandbox=...` and `--ro=/nrs/ml_datasets`:

```bash
# Bubblewrap, with your venv
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --sandbox=bwrap --ro=/nrs/ml_datasets \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=10 --batch-size=256 --save-models

# podman, with the image's own PyTorch and Ray (no --venv)
./submit_ray_job.sh 2 --queue=gpu_h200_parallel \
    --sandbox=podman --ro=/nrs/ml_datasets \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=10 --batch-size=256 --save-models
```

| Option | Meaning |
|---|---|
| `--sandbox=bwrap\|podman` | which sandbox |
| `--sandbox-dir=PATH` | where `sandbox-run.sh` / `podman-run.sh` live; default `$AGENTIC_SANDBOX_DIR`, else `PATH` |
| `--image=IMAGE` | podman only; default `agentic-sandbox-gpu` |
| `--ro=PATH`, `--rw=PATH` | extra directories the sandbox may read / write (repeatable). A `--rw` directory is created on every node if missing |

### What the sandbox sees

| Directory | Access | Who adds it |
|---|---|---|
| the directory you submit from (`ray/`) | read and write | the launcher |
| `../output` (job logs) and `../models` (checkpoints from `--save-models`) | read and write | the launcher |
| `/scratch/$USER/ray_results` (Ray Train's own results) | read and write | the launcher |
| your venv, with `bwrap` | read | the launcher |
| `/nrs/ml_datasets` (the datasets of every GPU example here) | read | you: `--ro=/nrs/ml_datasets` |
| anything else your own script reads or writes | | you: `--ro=` / `--rw=` |

Nothing else of your account is visible: no home directory, keys or tokens.

### The network

The launcher picks the network; there is nothing to choose:

| Job | Network inside the sandbox |
|---|---|
| 2+ nodes | the node's own network, not limited, so that the nodes reach each other. Files, home directory, keys and tokens stay hidden as always |
| 2+ nodes on an InfiniBand queue (B300, H200, H100) | the same, with the InfiniBand devices: NCCL's data goes over InfiniBand directly |
| one node | none; Ray runs on the sandbox's own loopback |

### How it differs from a normal run

- **One sandbox per node holds everything.** The head node's sandbox starts Ray, waits for the
  other nodes, runs the script and stops Ray. Each other node's sandbox joins and stays until
  its raylet exits. Without `--sandbox`, Ray daemons are left on each node and the script runs
  in the job script.
- **Ray's ports are fixed**, from 11000 up. A multi-node job has its nodes to itself on the
  parallel queues, so they do not collide with another job. The per-job ports and the per-job
  temp dir described under [Monitoring](#monitoring) belong to a run without `--sandbox`.
- **Ray still has no authentication** on two or more nodes (see
  [Known limitations](#known-limitations)): the sandbox uses the node's network, and the head
  port is 11001. Someone who attaches runs code inside the sandbox, with what the sandbox sees.
- **Only `submit_ray_job.sh` has `--sandbox`.** `submit_ray_cpu_job.sh` and `run_inference.sh`
  do not.

---

## Monitoring

Job output goes to `../output/` (relative to the directory you submit from; the submit scripts create it). At job end — and on startup failure, `bkill` or walltime — the submit scripts stop only that job's Ray processes and remove its per-job temp dir `/scratch/$USER/ray_$USER_<JOBID>` on every node; other Ray clusters you have on a shared node are left running. Each job also uses its own block of Ray ports (printed as `Per-job Ray ports` in the `.out` file):

```bash
bjobs                              # list jobs
tail -f ../output/*_<JOBID>.out    # training output
tail -f ../output/*_<JOBID>.err    # errors
```

---

## Known limitations

**The Ray cluster has no authentication.** The submit scripts start Ray with its defaults, so the head port (printed as `Head node: <host>  Port: <port>` in the job output) accepts connections from any machine that can reach the node — and compute nodes are directly reachable by hostname from the internal network. Anyone on the cluster network who finds that port can attach to the cluster and run code as you for as long as the job lives. The random port is not a protection.

What to do about it:

- Treat a running Ray job like an open shell on the cluster network: do not put secrets (tokens, passwords) in the job environment or script arguments, and do not run it on data that other cluster users must not read.
- Keep jobs no longer than needed. The cluster is torn down when the job ends, and `bkill <JOBID>` stops it at once.
- Recent Ray releases have built-in token authentication. It is not enabled here because it depends on the installed Ray version; check the Ray security documentation for your version and, if supported, enable it on every node of the job (head and workers) — this has not been tested with these scripts.
