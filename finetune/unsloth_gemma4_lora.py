#!/usr/bin/env python3
"""
Unsloth LoRA fine-tune of Gemma-4-26B-A4B on a single GPU.

Follows the Janelia "Fine-Tuning Gemma 4 26B-A4B" guide: **16-bit LoRA, not QLoRA**
(Gemma-4's MoE routing is unstable in 4-bit), scoped to the language-model layers only
(`finetune_vision_layers=False`) so the multimodal wrappers are left alone — this sidesteps
the Gemma4ClippableLinear day-zero bug. 16-bit base ≈ 52 GB resident → a single 80 GB-class GPU
(H100/H200/B300).

Unsloth gives ~1.5-2x speed and 50-70% less VRAM than a stock transformers+peft loop, with
identical resulting weights. Single-GPU only — for multi-node full fine-tuning (which stresses
the IB fabric) see the FSDP example.

Usage (via submit_unsloth.sh, or a single-GPU bsub):
    python unsloth_gemma4_lora.py \
        --model /misc/hf/google/gemma-4-26B-A4B-it \
        --dataset yahma/alpaca-cleaned --max-samples 1000 \
        --max-steps 60 --output ../models/gemma4_lora
"""
import argparse, os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/misc/hf/google/gemma-4-26B-A4B-it")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned",
                   help="a HF Hub dataset id, OR a local .jsonl/.json/.csv file. Either way, rows must "
                        "carry instruction/input/output (Alpaca) or prompt/response columns. This is the "
                        "data the adapter learns from — swap in your own examples for a real fine-tune")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="keep 0 — Gemma-4 MoE expert LoRA (lora.ParamWrapper) rejects nonzero dropout, "
                        "and dropout=0 is Unsloth's optimized fast path")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=60, help="use -1 to train full --epochs")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output", default="../models/gemma4_lora")
    args = p.parse_args()

    # import unsloth first (it patches transformers/trl for speed + lower VRAM)
    from unsloth import FastModel
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # ---- load base in 16-bit (NOT 4-bit — Gemma-4 MoE routing dislikes QLoRA) ----
    model, tok = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,          # 16-bit LoRA
        full_finetuning=False,
    )

    # ---- LoRA on the language layers only; leave the multimodal wrappers alone ----
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # Gemma-4 is vision+audio multimodal — don't touch those
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.lora_dropout,
        bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
    )
    model.print_trainable_parameters()

    # ---- dataset: a HF Hub id or a local .jsonl/.json/.csv — this is YOUR training data ----
    split = f"train[:{args.max_samples}]" if args.max_samples and args.max_samples > 0 else "train"
    if os.path.exists(args.dataset):
        builder = "csv" if args.dataset.rsplit(".", 1)[-1].lower() == "csv" else "json"
        ds = load_dataset(builder, data_files=args.dataset, split=split)
    else:
        ds = load_dataset(args.dataset, split=split)

    # ---- format each row into Gemma's chat template as a `text` field ----

    def to_text(ex):
        instr = ex.get("instruction", "") or ex.get("prompt", "")
        inp = ex.get("input", "") or ""
        out = ex.get("output", "") or ex.get("response", "") or ex.get("completion", "")
        user = instr if not inp else f"{instr}\n\n{inp}"
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": out}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    # ---- train (TRL SFTTrainer) ----
    cfg = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=5,
        bf16=True,
        optim="adamw_8bit",
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)

    stats = trainer.train()
    print(f"[unsloth] train loss: {stats.training_loss:.4f}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)          # LoRA adapters
    tok.save_pretrained(args.output)
    print(f"[unsloth] LoRA adapters saved to {args.output}")


if __name__ == "__main__":
    main()
