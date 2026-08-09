# gpu-cluster-recipes

Runnable recipes for the Janelia GPU cluster (LSF): distributed training with **Ray** and
**PyTorch/torchrun**, **fine-tuning** (LoRA and multi-node FSDP), and large-model inference with
**vLLM**. Each area is self-contained in its own directory with its own README.

**Cluster context:** jobs submit via LSF (`bsub`). GPU queues: `gpu_l4`, `gpu_a100`, `gpu_h100`,
`gpu_h200`, `gpu_b300` (plus `*_parallel` variants for whole-node / multi-node). Shared models
live under `/misc/hf/`, shared Apptainer images under `/misc/local/singularity/`.

## Contents

| Directory | What's there |
|---|---|
| [`ray/`](ray/) | Ray Train examples — CIFAR-10, GPT-2, ImageNet (ResNet-50 / ViT), LIVECell segmentation, protein sequence search — plus `submit_ray_job.sh` |
| [`torchrun/`](torchrun/) | Multi-node PyTorch launcher `submit_torchrun_job.sh`, a ViT example, and the NCCL/InfiniBand test suite |
| [`finetune/`](finetune/) | Fine-tuning — single-GPU LoRA (Unsloth, `submit_unsloth.sh`) and multi-node full FT (FSDP, `fsdp_sft.py`), plus a dense-vs-MoE + fine-tuning primer |
| [`nccl-ib/`](nccl-ib/) | Shared NCCL/InfiniBand env configs (`nccl_ib.sh`, `nccl_ib_b300.sh`), sourced by both launchers on IB queues |
| [`vllm/`](vllm/) | Serve any Hugging Face model with vLLM (Apptainer / venv / Podman) + optional Open WebUI — see the two guides |

## Quick starts

Ray training (run from `ray/`):

```bash
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- --num-gpus=16 --epochs=10
```

Serve a model with vLLM (auto-selects queue / GPUs / parsers):

```bash
vllm/serve_vllm.sh --model /misc/hf/openai/gpt-oss-20b            # normal model, auto placement
vllm/serve_vllm.sh --model /misc/hf/moonshotai/Kimi-K3 --webui   # full B300 + chat UI
```

Fine-tune a model with LoRA (single GPU, run from `finetune/`):

```bash
./submit_unsloth.sh --dataset=/path/to/mydata.jsonl --output=../models/gemma4_myadapter
```

NCCL / InfiniBand bandwidth test:

```bash
torchrun/submit_torchrun_job.sh 4 --queue=gpu_b300_parallel --venv=~/torch_env \
    --script=$PWD/torchrun/test_torchrun_ibstress.py -- --max-mb 4096 --secs 600
```

See each directory's `README.md` (and, for vLLM, `vllm/VLLM_SERVING_GUIDE.md` +
`vllm/KIMI_K3_VLLM_GUIDE.md`) for full details.

> **Job logs:** vLLM scripts write to `output/` at the repo root (git-ignored); the ray/torchrun
> submit scripts default to `../output` (override with `--output-dir`).
