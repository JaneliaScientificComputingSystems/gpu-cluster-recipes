# Dense vs Mixture-of-Experts: A Primer on Architecture and Fine-Tuning

An introduction for anyone wanting to understand modern large language models: what dense and
Mixture-of-Experts (MoE) architectures actually are, why MoE has become the default for frontier
models, how each is trained across many GPUs, and the full menu of fine-tuning approaches (full,
LoRA/QLoRA, and the rest) with the dense-vs-MoE differences called out.

No prior distributed-training background is assumed.

---

## 1. Background: where the parameters live

Every transformer layer is two sublayers:

1. **Self-attention** — mixes information across tokens in a sequence. Relatively few parameters
   (the Q/K/V/O projections).
2. **Feed-forward network (FFN / MLP)** — processes each token independently, usually
   `hidden → ~4×hidden → hidden`. This is **~2/3 of the model's parameters**.

Because the FFN dominates the parameter count, it's the part both dense and MoE architectures build
around. The whole distinction is in **how the FFN is structured and how much of it runs per token**.

---

## 2. Dense models

In a dense model, **every token passes through every parameter**. There is one FFN per layer and
all of it fires for all tokens.

```
token ──► attention ──► FFN (all weights) ──► next layer
```

- **Active params per token = total params.** A 70B dense model does 70B params of work per token.
- **Compute (FLOPs) ≈ 2 × total_params per token** (forward pass). Grows linearly with size.
- Simple, predictable, uniform: every unit of hardware does identical work, and communication
  patterns during training are regular.

**Examples:** the Llama 3.x series (8B–405B), Qwen2.5 (up to 72B), Mistral-Large (123B),
Falcon-180B, Nemotron-4 340B. **Llama-3.1-405B is the largest mainstream dense LLM** — nearly
everything bigger has moved to MoE.

---

## 3. Mixture-of-Experts (MoE) models

MoE replaces the single FFN with **N independent "expert" FFNs plus a small "router" (gate)**. For
each token, the router scores the experts and sends the token to only its **top-k** (e.g. 2 of 8,
or 4 of 128). The other experts don't run for that token.

```
                 ┌─► expert 1 ┐
token ─► router ─┼─► expert 2 ┤─► weighted sum of the top-k picks ─► next layer
  (top-k)        ├─► ...       │
                 └─► expert N ┘   (the rest stay idle for this token)
```

### 3.1 Total vs active parameters

This is the defining property of MoE:

- **Total params** — all experts must exist in memory. Huge.
- **Active params per token** — only the k selected experts compute. Small.

Model names often encode it: **`Qwen3-30B-A3B`** = 30B total, **A3B ≈ 3B active**. The **sparsity
ratio** (active/total) is commonly 1/8 to 1/30.

| Model | Total | Active/token | Experts (top-k) |
|---|---|---|---|
| Mixtral 8×7B | ~47B | ~13B | 8 (top-2) |
| Qwen3-30B-A3B | 30B | ~3B | 128 (top-8) |
| GPT-OSS-120B | 120B | ~5B | 128 (top-4) |
| DeepSeek-V3 | 671B | 37B | 256 + shared (fine-grained) |
| Qwen3-235B-A22B | 235B | 22B | 128 (top-8) |
| Llama-4-Maverick | ~400B | 17B | 128 |

### 3.2 The router and its failure modes

The router is a tiny learned linear layer: `scores = softmax(x · W_gate)`, then keep the top-k
experts and renormalize their weights. Training the router is where MoE gets delicate:

- **Load balancing.** Left alone, the router collapses to always using a few "favorite" experts —
  the rest never train (wasted capacity). Fixed with an **auxiliary load-balancing loss** that
  pushes token assignment toward uniform across experts. It's an *extra loss term you must tune*.
- **Router z-loss.** A stabilizer penalizing large gate logits; without it, routing logits can blow
  up and destabilize training.
- **Capacity factor & token dropping.** For efficient batched compute, each expert gets a fixed
  **capacity** (max tokens per batch). Overflow tokens are **dropped** (skip the FFN) or padded. Too
  low → dropped tokens hurt quality; too high → wasted compute. Another knob.
- **Low-precision fragility.** Routing is a discrete, near-argmax decision over close scores, so
  it's sensitive to numerical noise. In 4-bit, small quantization errors can flip expert
  assignments and destabilize training — which is why MoE fine-tuning usually avoids 4-bit
  quantization (see §7.4).

### 3.3 Refinements

- **Shared experts** (e.g. DeepSeek): one or more experts always run for every token, capturing
  common knowledge, while the routed experts specialize. Improves stability.
- **Fine-grained experts**: many small experts (256+) with a higher top-k, giving finer
  specialization for the same active-parameter budget.

---

## 4. Why MoE became the standard for frontier models

Most recent frontier open models — DeepSeek-V3, the larger Qwen3 variants, Llama 4, GPT-OSS,
Gemma-4 — are **MoE**, and GPT-4 is widely reported to be MoE as well. The reasons:

1. **Decouples capacity from compute (the core win).** Model *quality* scales with **total
   parameters** (knowledge capacity), but *cost* scales with **active parameters** (FLOPs per
   token). MoE lets you grow total params ~10× while keeping per-token compute roughly flat — a much
   more capable model for about the same training and inference FLOPs.

2. **Cheaper inference at a given quality.** Serving a 671B-total / 37B-active model costs about what
   a 37B dense model costs per token — but answers like something far larger. For high-volume serving
   that economics is decisive.

3. **Training-FLOP efficiency.** Under a fixed compute budget, a sparse MoE typically reaches lower
   loss than a dense model of equal *active* size, because it stores more knowledge per FLOP spent.

4. **Specialization.** Experts empirically specialize (syntax, code, math, languages), which helps
   on heterogeneous data.

### The costs MoE pays (and why dense isn't obsolete)

- **Memory footprint.** You still **store every expert** even though few run — a 671B MoE needs 671B
  params of memory regardless of the small active count. MoE trades compute for memory.
- **Communication.** Efficient MoE training needs **all-to-all** token routing between GPUs (§5), a
  harder, less regular collective than dense's all-gather/reduce-scatter.
- **Complexity & instability.** Router losses, capacity tuning, load imbalance, low-precision
  fragility (§3.2).
- **Fine-tuning is harder** (§7.4).

Dense models remain the right choice when you want **simplicity, predictable behavior, easy
fine-tuning, a small total-memory footprint, or a clean, reproducible workload**.

---

## 5. Training across many GPUs: parallelism strategies

Fine-tuning a large model means splitting the model and/or data across many GPUs. The strategy
determines the communication pattern between GPUs, which is the main performance factor at scale.

| Strategy | What it splits | Dominant collective | Notes |
|---|---|---|---|
| **DDP** (data parallel) | data (full model replicated per GPU) | all-reduce (grads) | Only works if the whole model + optimizer fits on one GPU. |
| **FSDP / ZeRO** | params, grads, optimizer state (sharded) | all-gather (fwd) + reduce-scatter (bwd) | The workhorse for large *dense* models. `FULL_SHARD` = ZeRO-3. |
| **Tensor parallel (TP)** | individual matmuls, within a layer | all-reduce per layer | Bandwidth-heavy; usually kept within a node (fast NVLink). |
| **Pipeline parallel (PP)** | layers across GPU groups | point-to-point (activations) | Introduces "bubble" overhead; good across nodes. |
| **Expert parallel (EP)** | MoE experts across GPUs | **all-to-all** (token dispatch/combine) | **MoE-only.** The reason MoE training differs. |
| **Sequence / context parallel** | the sequence dimension | all-gather / p2p | For very long context windows. |

Large-scale training composes several of these ("3D/4D parallelism": data × tensor × pipeline, plus
expert parallelism for MoE).

- **Dense** → typically **FSDP `FULL_SHARD`** (optionally + TP within a node). The communication
  volume is fixed and proportional to model size — predictable and easy to reason about.
- **MoE** → needs **EP** for the experts (all-to-all) *plus* FSDP/TP for the non-expert weights. The
  all-to-all volume depends on routing, batch composition, and sequence mix, so throughput is more
  variable and the system is more complex to tune.

### FSDP sharding modes (for dense models)

- **`FULL_SHARD`** (ZeRO-3): shard params + grads + optimizer across *all* GPUs; all-gather each
  layer's full weights just-in-time. Maximum memory savings, maximum communication.
- **`HYBRID_SHARD`**: shard within a node, replicate across nodes. Less cross-node traffic, more
  memory used per node.
- **`SHARD_GRAD_OP`** (ZeRO-2): shard grads + optimizer only, replicate params. Less communication,
  more memory. Choose the mode by trading memory against communication for your hardware.

---

## 6. Memory accounting: why size dictates GPU count

**Full fine-tuning with mixed-precision Adam ≈ 16 bytes per parameter**, before activations:

| Component | Bytes/param |
|---|---|
| bf16 parameters | 2 |
| bf16 gradients | 2 |
| fp32 master weights (optimizer) | 4 |
| fp32 Adam momentum `m` | 4 |
| fp32 Adam variance `v` | 4 |
| **Total optimizer + model state** | **16** |

Plus **activations** (proportional to batch × sequence × layers; reduced ~10× with activation
checkpointing) and **all-gather buffers** (transient room to reconstruct the largest layer's full
weights).

So a P-billion-parameter model needs roughly `16 × P` GB of sharded state:

| Model size | Full-FT state (~16 B/param) | Rough GPU count (80 GB-class) |
|---|---|---|
| 7B | ~110 GB | 2–4 GPUs |
| 32B | ~0.5 TB | ~8 GPUs (1 node) |
| 70B | ~1.1 TB | ~16 GPUs (2 nodes) |
| 405B | ~6.5 TB | ~64 GPUs (8 nodes) |

**MoE caveat:** use *total* params for the memory estimate (you store all experts) but *active*
params for the compute estimate. A 671B MoE needs 671B-worth of memory even though each token only
touches 37B.

**Parameter-efficient fine-tuning (§7.2) collapses this:** the frozen base is 2 bytes/param (bf16) or
~0.5 byte/param (4-bit), and only the tiny adapters carry the 16-bytes/param optimizer cost — so a
70B model can be LoRA-fine-tuned on a single 80 GB GPU.

---

## 7. Fine-tuning approaches

### 7.1 Full fine-tuning

Update **all** weights. Highest quality ceiling and most flexible, but ~16 bytes/param of state
(§6) → multi-GPU FSDP for anything beyond ~7B. Use it when you're substantially changing the model
(new domain or language, large high-quality dataset) and have the compute.

### 7.2 Parameter-efficient fine-tuning (PEFT)

Freeze the base model; train only a small set of new parameters. Far less memory, and the result is
a small, swappable **adapter** rather than a whole new copy of the model.

- **LoRA (Low-Rank Adaptation)** — the default. For a weight matrix `W (d×k)`, freeze it and learn a
  low-rank update `ΔW = (α/r) · B·A`, where `A (r×k)` and `B (d×r)` with rank `r ≪ min(d,k)`.
  Trainable parameters per matrix = `r·(d+k)` — often **well under 1%** of the model. Key knobs:
  **`r`** (capacity, typically 8–64), **`α`** (scaling, often `2r`), **dropout**, and **target
  modules** (which matrices get adapters — attention only, or attention + MLP).
- **QLoRA** — LoRA on top of a **4-bit quantized** frozen base (NF4 datatype + double quantization +
  paged optimizers). Cuts base memory ~4×, so a 70B model fits on a single 80 GB GPU.
  **Caveat: risky for MoE** — 4-bit noise destabilizes routing (§3.2).
- **DoRA** (weight-decomposed LoRA), **rsLoRA** (rank-stabilized scaling), **LoRA+** (different
  learning rates for `A` vs `B`) — incremental quality improvements over plain LoRA.
- **Other PEFT**: prefix/prompt tuning, `(IA)³`, BitFit (bias-only), classic bottleneck adapters.
  Mostly superseded by LoRA for LLMs, but useful in niche or very-low-data settings.

**Full vs LoRA rule of thumb:** LoRA to adapt *style / format / task* on modest data (hundreds to a
few thousand examples); full fine-tuning to teach *genuinely new knowledge or behavior* on large
data, when you can afford it.

### 7.3 Beyond SFT — the training objective also varies

Independent of full-vs-LoRA, *what* you optimize:

- **Continued pre-training** — more next-token prediction on a domain corpus (adapt knowledge).
- **SFT (supervised fine-tuning)** — train on instruction→response pairs. The most common starting
  point.
- **Preference optimization** — align to human/AI preferences: **DPO** (simple, offline, popular),
  **ORPO** (reference-free; folds SFT and preference into one step), **KTO**, **PPO** (classic RLHF,
  heaviest), **GRPO** (RL for reasoning). Libraries like TRL implement these, and they compose with
  either LoRA or full fine-tuning.

### 7.4 Dense-specific vs MoE-specific fine-tuning

**Dense** — straightforward. Full FT via FSDP, or LoRA/QLoRA on attention (± MLP). Behavior is
predictable and QLoRA is safe.

**MoE** — several extra considerations:

- **What to adapt.** The experts are the bulk of the parameters. Options: (a) LoRA the **attention +
  router + shared experts**, leaving the routed experts frozen (cheap, stable, common); (b) LoRA the
  experts too (many more adapters, more capacity, more complexity); (c) full-fine-tune everything
  (needs expert parallelism, §5).
- **Avoid 4-bit (QLoRA) by default.** Routing instability in 4-bit (§3.2) — prefer 16-bit LoRA.
- **Keep the router healthy.** Aggressive fine-tuning can wreck the learned load balance; sometimes
  the router is **frozen**, or the load-balancing loss is retained during fine-tuning.
- **Expert load imbalance** makes throughput lumpy and complicates sharding.
- **Upcycling** — a related technique: initialize an MoE by *cloning* a trained dense FFN into N
  experts, then continue training. This *creates* an MoE from a dense model rather than fine-tuning
  an existing one.

---

## 8. Decision guide

| Your goal | Model | Suggested approach |
|---|---|---|
| Adapt style/format/task, modest data, 1 GPU | any up to ~70B | **LoRA** (16-bit; QLoRA if dense and VRAM-limited) |
| Teach new knowledge/behavior, large data | dense up to ~70B | **Full FT** with FSDP `FULL_SHARD`, multi-GPU |
| Fine-tune a large MoE | MoE | **16-bit LoRA** on attention + router + shared experts |
| Align to human/AI preferences | any | **DPO / ORPO** (+ LoRA or full FT) |
| Train the very largest models | 100B+ dense or any MoE | Multi-node with composed parallelism (FSDP/TP/PP, + EP for MoE) |

---

## 9. Further reading

- Shazeer et al. 2017, *Outrageously Large Neural Networks* — the sparsely-gated MoE layer.
- Fedus et al. 2021, *Switch Transformers* — simplified top-1 routing and scaling.
- Rajbhandari et al. 2020, *ZeRO* — the sharded-optimizer memory math behind FSDP.
- Hu et al. 2021, *LoRA*; Dettmers et al. 2023, *QLoRA*.
- The DeepSeek-V3 and Mixtral technical reports — shared/fine-grained experts and modern MoE at
  scale.
