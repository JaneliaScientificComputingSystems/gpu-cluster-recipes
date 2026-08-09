# finetune — fine-tuning LLMs (LoRA and multi-node FSDP)

Two fine-tuning recipes that share one dataset format (see [The dataset](#the-dataset)):

1. **Single-GPU LoRA** with [Unsloth](https://github.com/unslothai/unsloth), worked end-to-end on
   **Gemma-4-26B-A4B** — fast, low-VRAM adapter training on one GPU.
2. **Multi-node full fine-tuning** with **PyTorch FSDP `FULL_SHARD`**, worked on the dense
   **Qwen2.5-72B-Instruct** — full-parameter training sharded across many GPUs/nodes.

Unsloth patches transformers/TRL with fused Triton kernels for ~1.5–2× speed and 50–70% less VRAM
than a stock `transformers`+`peft` loop. Companion to the Janelia guide
[*Fine-Tuning Gemma 4 26B-A4B — Hardware and Tooling*](https://hpc.int.janelia.org/docs/fine-tuning-gemma-4-26b-a4b-hardware-and-tooling-guide).

| File | What it is |
|---|---|
| `unsloth_gemma4_lora.py` | Single-GPU **LoRA** training script — 16-bit LoRA on the language layers |
| `submit_unsloth.sh` | LSF submit wrapper for the LoRA example (single GPU, caches on `/scratch`) |
| `fsdp_sft.py` | Multi-node **full fine-tune** training script — FSDP `FULL_SHARD` (dense LLM) |
| [`DENSE_VS_MOE_FINETUNING.md`](DENSE_VS_MOE_FINETUNING.md) | Reference: dense vs MoE architecture, why MoE is now standard, parallelism, and full/LoRA/QLoRA fine-tuning approaches |

---

## What you actually build

Fine-tuning here produces a **LoRA adapter** — a small set of extra weights (~494 MB for Gemma-4,
1.9% of the model). The 26B base model stays **frozen**; only the adapter is trained. So the one
thing *you* supply is the **training data** — the adapter is nothing more than the learned imprint
of your examples. Everything else (base model, LoRA machinery) is fixed.

> **Why 16-bit LoRA and not 4-bit QLoRA?** Gemma-4 is a mixture-of-experts model and its
> expert-routing is numerically unstable in 4-bit, which degrades adapter quality. We load the base
> in 16-bit (≈52 GB resident) and train LoRA on top — it still fits on one 80 GB GPU. LoRA is also
> scoped to the **language layers only** (`finetune_vision_layers=False`), leaving the multimodal
> wrappers untouched (this also sidesteps a Gemma-4 day-zero LoRA bug in the vision path).

---

## The dataset (this is the important part)

A fine-tuning dataset is a list of **input → desired-output** examples written in your domain's
voice. **Both scripts (LoRA and FSDP) use the identical format** — one of two column shapes per row:

- **Alpaca style:** `instruction`, `input` (optional), `output`
- **Chat style:** `prompt`, `response`

Provide it as a **local `.jsonl`** (one JSON object per line) or a **Hugging Face Hub dataset id** —
either way, pass it via `--dataset`. Each row is wrapped in the model's own chat template (Gemma's
`<start_of_turn>…`, Qwen's `<|im_start|>…`, etc. — applied automatically) and the model is trained to
produce the `output`/`response` given the `instruction`/`prompt`.

### Standard example dataset — `yahma/alpaca-cleaned`

The default is [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned), the
canonical generic instruction-tuning set (~52k rows). It downloads automatically from the Hugging
Face Hub on first use — nothing to stage. Each row looks like:

```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet...\n2. Exercise regularly...\n3. Get enough sleep..."
}
```

Use it as-is to prove the pipeline. It teaches generic instruction-following — useful as a template,
not as a real specialization.

### Bringing your own data

For a *real* adapter, supply your own examples in the **same format** as a local `.jsonl` (one JSON
object per line) and point `--dataset` at the file:

```json
{"instruction": "Classify this HPC ticket as hardware, software, quota, or access.", "input": "job died with CUDA error 802 on a B300", "output": "hardware"}
{"instruction": "Rewrite this lab note as a numbered protocol.", "input": "fixed cells 10 min PFA then washed 3x pbs", "output": "1. Fix in 4% PFA, 10 min.\n2. Wash 3× in PBS."}
```

```bash
./submit_unsloth.sh --dataset=/path/to/mydata.jsonl --output=../models/gemma4_myadapter
```

**Rules of thumb:** ~200–500 examples shift tone/format; ~1–5k build a real skill. Consistency
matters more than volume — the model copies whatever patterns are *most consistent* in your data.
Keep a ~10% held-out slice you don't train on to sanity-check the adapter afterward.

---

## Environment (one-time)

Unsloth needs a recent stack (transformers 5.x / TRL 0.24 / PyTorch 2.11). Build a venv with `uv`:

```bash
uv venv ~/unsloth_env --python 3.12
source ~/unsloth_env/bin/activate
uv pip install unsloth
```

This pulls unsloth + unsloth-zoo, transformers, peft, trl, torch (CUDA build), triton, bitsandbytes,
etc. (Note: `import unsloth` requires a GPU — it will error on the login node. Verify inside a job.)

---

## Run it

```bash
# Smoke test — tiny slice, 5 steps (proves load + LoRA + train + save)
./submit_unsloth.sh --max-samples=100 --max-steps=5

# Real adapter — full pass over your data on an H200
./submit_unsloth.sh --queue=gpu_h200 --dataset=/path/to/mydata.jsonl \
    --output=../models/gemma4_myadapter -- --epochs=3 --max-steps=-1
```

Options (see `./submit_unsloth.sh --help`): `--queue` (default `gpu_h100`), `--dataset`, `--output`,
`--model`, `--max-samples`, `--max-steps`, `--epochs`, `--walltime`, `--venv`.

**Queue guidance:** Gemma-4-26B in 16-bit needs an 80 GB-class GPU — `gpu_h100` (recommended),
`gpu_h200`, or `gpu_b300`. Fine-tuning is single-GPU (`num=1`); the wrapper puts `HF_HOME`,
`TMPDIR`, and `XDG_CACHE_HOME` on node-local `/scratch` since the shared trees are read-only for
caching.

> ✅ Validated 2026-08-08 (1× B300): loads the 52 GB base, applies LoRA to the language layers
> (**494M trainable params = 1.88%** of 26.3B), trains, and saves adapters cleanly.

---

## Using the adapter

`model.save_pretrained(output)` writes the LoRA adapter (`adapter_config.json` + safetensors), not a
full model. To use it, load the base and apply the adapter:

```python
from unsloth import FastModel
model, tok = FastModel.from_pretrained("/misc/hf/google/gemma-4-26B-A4B-it", load_in_4bit=False)
model.load_adapter("../models/gemma4_myadapter")
```

To serve it with vLLM, either pass the adapter as a LoRA module or first merge it into the base
(`model.save_pretrained_merged(...)`) and serve the merged weights — see [`../vllm/`](../vllm/).

---

## Multi-node full fine-tuning (FSDP)

Unsloth/LoRA above is single-GPU. To **full-fine-tune** a model too large to hold (with optimizer
state) on one GPU, `fsdp_sft.py` shards params + gradients + optimizer state across every GPU with
PyTorch **FSDP `FULL_SHARD`** (ZeRO-3) and runs across nodes over InfiniBand. Default target is the
dense **Qwen2.5-72B-Instruct** (~1.1 TB of training state → ~2 nodes minimum). Dense on purpose —
FSDP full-shard composes cleanly with a dense model; a large MoE would also need expert parallelism
(see [`DENSE_VS_MOE_FINETUNING.md`](DENSE_VS_MOE_FINETUNING.md)).

It's a HuggingFace `TRL` `SFTTrainer` job configured for FSDP, launched with the shared multi-node
launcher in [`../torchrun/`](../torchrun/) (which sets `RANK`/`WORLD_SIZE`/`LOCAL_RANK` and the
InfiniBand NCCL env — HF Trainer reads them automatically):

> ✅ Validated 2026-08-08 (2 nodes / 16× B300): Qwen2.5-72B loads, FSDP `FULL_SHARD` trains (loss
> drops cleanly), and rank 0 saves a complete, loadable HF checkpoint (6 safetensors shards +
> config/tokenizer). Point `--output` at a **shared, persistent** path (the default `../models/…`
> lives on `/groups`) — node-local `/scratch` is per-node and ephemeral. The full save is fp32, so
> the checkpoint is ~2× the bf16 base size.

```bash
# from this finetune/ directory — 2 nodes / 16 GPUs, quick 50-step validation
../torchrun/submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --venv=~/unsloth_env \
    --job-name=fsdp_sft --workdir=$PWD --script=$PWD/fsdp_sft.py -- \
    --model /misc/hf/Qwen/Qwen2.5-72B-Instruct --max-steps 50 \
    --output ../models/qwen72b_fsdp_sft

# full run — whole dataset over 4 nodes
../torchrun/submit_torchrun_job.sh 4 --queue=gpu_h200_parallel --venv=~/unsloth_env \
    --job-name=fsdp_sft --workdir=$PWD --script=$PWD/fsdp_sft.py -- \
    --model /misc/hf/Qwen/Qwen2.5-72B-Instruct --epochs 1 --lr 1e-5
```

Key script options: `--model`, `--dataset` (same [dataset format](#the-dataset) as the LoRA example
— HF Hub id or local `.jsonl`), `--layer-cls` (the transformer
block to FSDP-wrap — `Qwen2DecoderLayer` for Qwen2.5, `LlamaDecoderLayer` for Llama, etc.),
`--batch-size` (per-GPU micro-batch), `--grad-accum`, `--max-seq-len`, `--max-steps`/`--epochs`,
`--attn` (`sdpa` default; `flash_attention_2` if installed). Effective batch =
`batch-size × grad-accum × world-size`.

The **same script scales to larger dense models** by swapping `--model` (and `--layer-cls` if the
architecture differs) and adding nodes — e.g. `meta-llama/Llama-3.1-405B` genuinely needs ~8 nodes.

> Full-parameter FT is bandwidth-heavy: `FULL_SHARD` all-gathers every layer's weights each forward
> and reduce-scatters gradients each backward, across nodes — so this doubles as a real-world
> InfiniBand workload. Serve the result with [`../vllm/`](../vllm/) (`serve_vllm.sh` knows
> Qwen2.5-72B).
