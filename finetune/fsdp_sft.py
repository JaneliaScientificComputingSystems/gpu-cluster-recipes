#!/usr/bin/env python3
"""
Multi-node full fine-tuning of a large DENSE LLM with PyTorch FSDP.

Full-parameter SFT of a big dense model (default Qwen2.5-72B-Instruct) across
multiple nodes using Fully Sharded Data Parallel (FSDP `FULL_SHARD` / ZeRO-3):
parameters, gradients, and optimizer state are all sharded across every GPU, and
each transformer layer's full weights are all-gathered just-in-time for the
forward pass. This is the standard way to full-fine-tune a model too large to
hold (with optimizer state) on a single GPU — a 72B model needs ~1.1 TB of state,
so ~2 nodes of 8x80GB GPUs at minimum.

Dense on purpose: FSDP full-shard composes cleanly with a dense model. A large
MoE would additionally need expert parallelism (all-to-all) — see
DENSE_VS_MOE_FINETUNING.md.

Launch with the multi-node torchrun launcher, which sets RANK/WORLD_SIZE/LOCAL_RANK
(HuggingFace Trainer reads them automatically) — see the README:

    ../torchrun/submit_torchrun_job.sh 2 --queue=gpu_h200_parallel --venv=~/unsloth_env \
        --job-name=fsdp_sft --workdir=$PWD --script=$PWD/fsdp_sft.py -- \
        --model /misc/hf/Qwen/Qwen2.5-72B-Instruct --max-steps 50
"""
import os

# Caches/temp on node-local /scratch, set before importing HF libs. We FORCE these
# (not setdefault) to override any inherited shared-NFS HF cache — otherwise every rank
# hammers one shared cache dir. The datasets cache is additionally PER-RANK so concurrent
# load_dataset() calls across ranks can't race on the same cache files (a shared-cache race
# otherwise crashes multi-rank jobs with "Failed to open ... .arrow" / ".incomplete").
_user = os.environ.get("USER", "user")
_rank = os.environ.get("RANK", "0")
_base = f"/scratch/{_user}"
os.environ["HF_HOME"] = f"{_base}/hf_home"
os.environ["HF_HUB_CACHE"] = f"{_base}/hf_home/hub"
os.environ["HF_DATASETS_CACHE"] = f"{_base}/hf_datasets/rank{_rank}"
os.environ["TMPDIR"] = f"{_base}/tmp"
os.environ["XDG_CACHE_HOME"] = f"{_base}/xdg_cache"
for _d in (os.environ["HF_HOME"], os.environ["HF_HUB_CACHE"],
           os.environ["HF_DATASETS_CACHE"], os.environ["TMPDIR"], os.environ["XDG_CACHE_HOME"]):
    os.makedirs(_d, exist_ok=True)

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/misc/hf/Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned",
                   help="HF Hub id or local .jsonl/.csv (instruction/input/output or prompt/response)")
    p.add_argument("--max-samples", type=int, default=0, help="0 = use the whole dataset")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1, help="per-GPU micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = train full --epochs")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--layer-cls", default="Qwen2DecoderLayer",
                   help="transformer block class to FSDP-wrap (model-specific; e.g. "
                        "Qwen2DecoderLayer for Qwen2.5, Qwen3DecoderLayer for Qwen3, "
                        "LlamaDecoderLayer for Llama)")
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--output", default="../models/qwen72b_fsdp_sft")
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # bf16 base; low_cpu_mem_usage so host RAM isn't spiked before FSDP shards the model.
    # use_cache=False is required with gradient checkpointing.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn, low_cpu_mem_usage=True, use_cache=False,
    )

    # ---- dataset (HF Hub id or local file) → chat-templated "text" field ----
    split = f"train[:{args.max_samples}]" if args.max_samples and args.max_samples > 0 else "train"
    if os.path.exists(args.dataset):
        builder = "csv" if args.dataset.rsplit(".", 1)[-1].lower() == "csv" else "json"
        ds = load_dataset(builder, data_files=args.dataset, split=split)
    else:
        ds = load_dataset(args.dataset, split=split)

    def to_text(ex):
        instr = ex.get("instruction", "") or ex.get("prompt", "")
        inp = ex.get("input", "") or ""
        out = ex.get("output", "") or ex.get("response", "") or ex.get("completion", "")
        user = instr if not inp else f"{instr}\n\n{inp}"
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": out}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    cfg = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        dataset_text_field="text",
        max_length=args.max_seq_len,   # raw TRL 0.24 arg name (Unsloth patches it to max_seq_length)
        report_to="none",
        ddp_timeout=3600,   # generous: 16 ranks each read ~140 GB from NFS before the first collective
        # ---- FSDP FULL_SHARD (ZeRO-3): shard params+grads+optimizer across all ranks ----
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": [args.layer_cls],
            "backward_prefetch": "backward_pre",   # overlap next all-gather with backward
            "forward_prefetch": True,
            "use_orig_params": True,
            "limit_all_gathers": True,
            "sync_module_states": True,            # identical init across ranks
            "state_dict_type": "FULL_STATE_DICT",  # gather to one loadable checkpoint at save
        },
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    if is_main:
        print(f"[fsdp] world_size={world} | model={args.model} | FSDP FULL_SHARD | "
              f"per-GPU bs={args.batch_size} x grad_accum={args.grad_accum} "
              f"x world={world} = effective batch {args.batch_size * args.grad_accum * world}")

    trainer.train()

    # FULL_STATE_DICT gathers the sharded weights into one checkpoint on rank 0.
    trainer.save_model(args.output)
    if is_main:
        tok.save_pretrained(args.output)
        print(f"[fsdp] full model saved to {args.output}")

    # Rank 0's full-checkpoint write (~140 GB for 72B) takes minutes while other ranks
    # would otherwise race to torchelastic's exit barrier and trip its 300 s timeout —
    # so hold everyone here until the write finishes (uses the 3600 s process-group timeout).
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
