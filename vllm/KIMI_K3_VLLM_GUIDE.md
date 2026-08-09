# Serving Kimi-K3 with vLLM on a full B300 node

Kimi-K3-specific companion to the general **[VLLM_SERVING_GUIDE.md](VLLM_SERVING_GUIDE.md)**.
Everything shared (engines, one-time setup, clients, Open WebUI, troubleshooting) lives in the
general guide; this page covers only what's different for Kimi-K3.

---

## What makes Kimi-K3 special

| | |
|---|---|
| Model | **Kimi-K3** (`/misc/hf/moonshotai/Kimi-K3`) |
| Type | Mixture-of-Experts, **2.8 T total / 104 B active**, 896 experts, KDA + Gated MLA |
| Precision | **MXFP4 weights / MXFP8 activations** (Blackwell-native) — **1.5 TB on disk** |
| Context / modality | up to **1,048,576** tokens; multimodal (text + image, MoonViT-V2) |
| Hardware | **one full B300 node = 8 GPUs**, tensor-parallel 8 (~187 GiB weights/GPU) |
| vLLM | **≥ 0.27.0** (day-0 support) — **container-only** for now |

Three things differ from a normal model:

1. **Dedicated image (not the generic SIF).** Kimi-K3 needs vLLM ≥ 0.27.0 with CUDA-13
   MXFP4/FlashKDA kernels, which ship **only** in the vendor image — there's no PyPI wheel yet
   (stable maxes at 0.26.0), so **the venv path does not work for Kimi**. Use:
   - shared SIF: **`/misc/local/singularity/vllm-kimi-k3.sif`**, or
   - build your own: `apptainer pull … docker://vllm/vllm-openai:kimi-k3` (cu130; needs r580+
     driver — our B300s are r610/CUDA-13.3). Podman image: `docker.io/vllm/vllm-openai:kimi-k3`.
2. **Whole B300 node** (`--full-node`): 1.5 TB of weights need all 8 GPUs' HBM.
3. **Tool + reasoning parsers are `kimi_k3`**, and Kimi *always* reasons (returns
   `reasoning_content`).

Official recipe: <https://recipes.vllm.ai/moonshotai/Kimi-K3>

---

## The one command

Apptainer (preferred), full node, tool calling + reasoning, chat UI, kept up for 14 days:
```bash
./serve_vllm.sh --model /misc/hf/moonshotai/Kimi-K3 \
    --sif /misc/local/singularity/vllm-kimi-k3.sif --full-node \
    --enable-auto-tool-choice --tool-parser kimi_k3 --reasoning-parser kimi_k3 \
    --walltime 336:00 --webui
```

Podman variant (vendor image, GPU via CDI):
```bash
./serve_vllm.sh --model /misc/hf/moonshotai/Kimi-K3 --engine podman \
    --image docker.io/vllm/vllm-openai:kimi-k3 --full-node \
    --enable-auto-tool-choice --tool-parser kimi_k3 --reasoning-parser kimi_k3
```

> ⚠️ **Podman requires a one-time HPC setup first** — rootless storage on `/scratch`
> (`~/.config/containers/storage.conf` + `containers.conf`) and **CDI** for GPU access
> (`nvidia-ctk cdi generate`, admin/root). Do it once following the general guide's
> **§3c ("Podman (rootless) — one-time HPC config")** in
> [VLLM_SERVING_GUIDE.md](VLLM_SERVING_GUIDE.md) **before** the command above.
> **Apptainer (the recommended path above) needs none of this** — that's the main reason to prefer it.

> Drop `--webui` to serve the API only; add `--exit-after-test` to just measure load time and quit.
> The `--tool-parser/--reasoning-parser` flags are **required** if you'll use Open WebUI or any
> tool-calling client, or chat fails with *`"auto" tool choice requires …`*.

---

## Verified load timings (full B300, max-model-len 32768)

| run | image/SIF prep | weight load | launch → serving |
|---|---|---|---|
| **podman**, cold node | 107 s (pull 25 GB) | 759 s | **1382 s (~23 min)** |
| **apptainer**, cold node | 0 s (SIF pre-staged) | 565 s | **1221 s (~20 min)** |
| **podman**, warm re-serve (same node) | 1 s (cached) | fast (page cache) | **756 s (~13 min)** |

Notes: Apptainer wins cold-start (no 25 GB pull). A warm node (image cached in `/scratch`,
weights still in the ~3.9 TB page cache) roughly halves startup. The 1.5 TB NFS read is
front-loaded (first ~16/96 shards slow, then ~1.6 s/shard), and after weights load vLLM spends a
few minutes autotuning FP4-MoE kernels + capturing CUDA graphs before `/health` goes green.

---

## Talking to Kimi-K3

Standard OpenAI API (model id `kimi-k3`) — see the general guide §6 for curl/Python. One
Kimi-specific quirk:

> **Preserved-thinking:** Kimi-K3 always returns a separate `reasoning_content`. For multi-turn
> and tool calls, pass the assistant message back **verbatim** (including `reasoning_content` and
> any `tool_calls`), not just `content`.

For the browser UI, `--webui` (above) or `serve_openwebui.sh --model-url http://<node>.int.janelia.org:8000/v1`.

---

## Everything else

Engines & one-time setup, clients, Open WebUI details, SSH tunnels, tuning, and the full
troubleshooting table (including the `/nrs` `--cleanenv` fix, the bsub slot-ratio hang, CDI, and
Fabric-Manager CUDA-802) are in **[VLLM_SERVING_GUIDE.md](VLLM_SERVING_GUIDE.md)**.
