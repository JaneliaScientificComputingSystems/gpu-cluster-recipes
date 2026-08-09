# Serving LLMs with vLLM on the Janelia LSF cluster

A complete, from-scratch guide to serving **any** Hugging-Face model (staged under
`/misc/hf`) with **vLLM** on the Janelia GPU cluster — via **Apptainer** (preferred),
a **Python venv**, or **rootless Podman** — and talking to it (curl / Python / a browser
chat UI). One launcher, `serve_vllm.sh`, drives all of it.

> For **Kimi-K3** specifically (2.8T MoE, whole B300 node, special image) see the companion
> **[KIMI_K3_VLLM_GUIDE.md](KIMI_K3_VLLM_GUIDE.md)** — it reuses everything here and only adds
> the Kimi-specific bits.

---

## 1. TL;DR

**Preferred path — Apptainer with the shared image, zero setup.** `serve_vllm.sh` defaults to
Apptainer and auto-selects the queue/GPUs/parsers from the model path:

```bash
# serve a model (auto: picks queue, GPU count, and tool/reasoning parsers):
./serve_vllm.sh --model /misc/hf/openai/gpt-oss-20b

# ...or serve + a browser chat UI, printing its URL:
./serve_vllm.sh --model /misc/hf/Qwen/Qwen3-32B --webui
```

**Engine order of preference: Apptainer → venv → Podman.**
- **Apptainer** (default) — no per-user setup, uses the shared SIF (§3a).
- **venv (`--engine venv`)** — a Python env for sub-node models; one-time `uv` setup (§3b).
- **Podman (`--engine podman`)** — only for whole-node jobs wanting the exact vendor image;
  one-time HPC setup + CDI (§3c).

Everything below explains hardware sizing, the three engines, tool calling, clients, the UI,
and troubleshooting.

---

## 2. Pick your hardware (smallest GPU that fits)

Rule of thumb for a serving footprint with headroom for **1–2 concurrent users**:

```
VRAM needed  ≈  weight size (≈ the model's on-disk size)  +  ~20–30% for KV cache & activations
```

Choose the **smallest** GPU (or fewest GPUs) whose **total** memory clears that. Use
tensor-parallel (`--gpus N`, `--tp N`) when one GPU isn't enough.

**GPU memory per class (approx):** L4 = 24 GB · A100 = 40 or 80 GB · H100 = 80 GB ·
H200 = 141 GB · **B300 ≈ 268 GB** (275 GB reported).

**Recommended placement for the staged models** (weights ≈ on-disk; verify against vLLM's
`Model loading took … GiB` log line and raise if you need long context / more users):

| Model | On disk | Smallest sensible placement | Queue |
|---|---|---|---|
| `openai/gpt-oss-20b` | 39 G | 1× A100-80 / H100 | `gpu_h100` / `gpu_a100` |
| `google/gemma-4-26B-A4B-it` | 49 G | 1× A100-80 / H100 | `gpu_h100` |
| `Qwen/Qwen3-30B-A3B` | 57 G | 1× H100 (tight KV) or 1× H200 | `gpu_h100` / `gpu_h200` |
| `Qwen/Qwen3-32B` | 62 G | 1× H200, or 2× H100/A100 | `gpu_h200` |
| `Qwen/Qwen2.5-72B-Instruct` | ~145 G | 2× H200, or 1–2× B300 | `gpu_h200` / `gpu_b300` |
| `openai/gpt-oss-120b` | 183 G | 2× H200, or 1× B300 | `gpu_h200` / `gpu_b300` |
| `mistralai/Mistral-Small-4-119B-2603` | 226 G | 2× H200, or 1–2× B300 | `gpu_h200` / `gpu_b300` |
| `moonshotai/Kimi-K3` | 1.5 T | **8× B300 (whole node)** | `gpu_b300` (see Kimi guide) |

> These are starting points. Small MXFP4 models (e.g. gpt-oss-20b) use *less* live VRAM than
> their disk size suggests, so they may fit a smaller GPU; long context needs *more* (KV cache).

> ✅ **Validated (2026-08-08):** all of the above served cleanly at the auto-selected placement —
> gpt-oss-20b / Qwen3-30B-A3B / gemma-4 on 1× H100; Qwen3-32B on 1× H200; gpt-oss-120b &
> Mistral-Small-4 on 2× H200 — loading in ~200–360 s, with chat working and tool calling confirmed
> for gemma-4 (`gemma4`), Qwen3 (`hermes`) and Mistral (`mistral`).

**Node sharing:** models that need only 1–few GPUs can share a node with other jobs — request
just what you need (`--gpus N`, default `exclusive_process`; others use the node's remaining
GPUs). Only whole-node models (Kimi) reserve everything.

---

## 3. One-time setup — pick an engine

Decision guide:

| Situation | Engine |
|---|---|
| Anything, simplest, no per-user setup, portable | **Apptainer** (preferred) |
| Sub-node model, want a plain Python env, quick iteration | **venv (uv)** |
| Whole-node model where you want the exact vendor image | **Podman** (rootless) |

> **Podman only for whole-node reservations.** Rootless Podman has no cgroup resource
> containment here, so it must own the node. For shared/sub-node models use Apptainer or venv.

### 3a. Apptainer (preferred)

Use the shared generic image — **nothing to install**:
```
/misc/local/singularity/vllm-openai-v0.26.0.sif
```
Or build your own (e.g. a newer vLLM). `/scratch` is purged, so build there but **move the
`.sif` off scratch**:
```bash
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer pull /scratch/$USER/vllm-openai-v0.26.0.sif docker://vllm/vllm-openai:v0.26.0
mv /scratch/$USER/vllm-openai-v0.26.0.sif $HOME/     # persist off scratch
```
GPUs come via `--nv` — **no CDI needed**. (`serve_vllm.sh --engine apptainer` handles the rest.)

### 3b. venv (uv)

The nodes' system Python is 3.9 (too old) and there's no python module — use **`uv`** (it
fetches a standalone 3.12):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have uv
export PATH="$HOME/.local/bin:$PATH"
uv venv $HOME/vllm_env --python 3.12 --seed
uv pip install --python $HOME/vllm_env/bin/python vllm==0.26.0
```
Good for models that fit on 1–few GPUs and share a node. (`serve_vllm.sh --engine venv --venv …`.)

### 3c. Podman (rootless) — one-time HPC config

```bash
mkdir -p ~/.config/containers /scratch/$USER/podman-run /scratch/$USER/podman-storage
cat > ~/.config/containers/storage.conf <<'EOF'
[storage]
driver = "overlay"
runroot = "/scratch/$USER/podman-run"
graphroot = "/scratch/$USER/podman-storage"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
EOF
cat > ~/.config/containers/containers.conf <<'EOF'
[engine]
cgroup_manager = "cgroupfs"
events_logger = "file"
[network]
network_backend = "cni"
EOF
cat > ~/.config/containers/registries.conf <<'EOF'
unqualified-search-registries = ["docker.io"]
EOF
echo 'unset XDG_RUNTIME_DIR' >> ~/.bashrc && unset XDG_RUNTIME_DIR
podman system reset -f
```
GPU access needs **CDI**, generated once per node by an admin (root):
`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. `serve_vllm.sh` runs the required
per-job `podman system migrate` hygiene automatically.

---

## 4. Serving with `serve_vllm.sh`

```bash
# smallest model, one H100, apptainer + shared SIF:
./serve_vllm.sh --model /misc/hf/openai/gpt-oss-20b --queue gpu_h100 --gpus 1

# bigger model across 2 H200:
./serve_vllm.sh --model /misc/hf/openai/gpt-oss-120b --queue gpu_h200 --gpus 2

# via a venv instead of a container:
./serve_vllm.sh --model /misc/hf/Qwen/Qwen3-32B --engine venv --venv $HOME/vllm_env \
    --queue gpu_h200 --gpus 1

# measure load time then exit:
./serve_vllm.sh --model /misc/hf/google/gemma-4-26B-A4B-it --queue gpu_h100 --gpus 1 --exit-after-test
```

Key options: `--engine apptainer|podman|venv` (default apptainer) · `--model` (required) ·
`--sif`/`--image`/`--venv` · `--gpus`/`--tp` · `--full-node` · `--queue` ·
`--gpu-mode exclusive_process|shared` · `--max-model-len` (32768) · `--gpu-mem-util` (0.90) ·
`--enable-auto-tool-choice` · `--tool-parser` · `--reasoning-parser` · `--walltime` ·
`--host-pin` · `--exit-after-test` · `--webui` (§7). Run `./serve_vllm.sh --help` for all.

The script picks CPU slots to match the queue's ratio (12/GPU, 8/GPU on L4), times the load,
waits for `/health`, and smoke-tests. The server binds `0.0.0.0:8000` → reachable at
`http://<node>:8000/v1`.

> ⚠️ **Walltime:** omitting `-W` gives the queue *default* (often 120 min) and the job dies at
> 2 h. Pass a large `--walltime` (e.g. `336:00` = 14 days) to keep a server up.

---

## 5. Tool calling & reasoning

Chat UIs (and many agent clients) send `tool_choice: "auto"`, which vLLM **rejects** unless the
server was started with tool calling enabled — you'll see
*`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`*.

Enable it with the **model-specific** parsers:
```bash
--enable-auto-tool-choice --tool-parser <NAME> --reasoning-parser <NAME>
```

`serve_vllm.sh` **auto-applies** these for the staged models (registry) — you only set them by
hand for a model it doesn't know:

| Model | `--tool-parser` | `--reasoning-parser` |
|---|---|---|
| Kimi-K3 | `kimi_k3` | `kimi_k3` |
| Qwen3-32B / Qwen3-30B-A3B | `hermes` | `qwen3` |
| Qwen2.5-72B-Instruct | `hermes` | — (not a reasoning model) |
| Gemma-4-26B-A4B | `gemma4` | — (not a reasoning model) |
| Mistral-Small-4-119B | `mistral` | — |
| gpt-oss-120b / 20b | harmony format — set manually; see vLLM tool-calling docs | — |

If unsure of a parser name, check the image: `apptainer exec --nv <sif> vllm serve --help`
lists the valid choices, or see <https://docs.vllm.ai/en/stable/features/tool_calling/>.

---

## 6. Talking to the server

The API is OpenAI-compatible; the model id is whatever `--served-name` is (defaults to the
model dir basename, e.g. `gpt-oss-20b`).

**curl:**
```bash
NODE=<compute node>       # or 127.0.0.1 if tunneled / on-node
curl -s http://$NODE:8000/v1/models | python3 -m json.tool
curl -s http://$NODE:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Explain InfiniBand in one sentence."}],"max_tokens":256}' \
  | python3 -m json.tool
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://<node>:8000/v1", api_key="dummy")  # vLLM ignores the key
r = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain InfiniBand in one sentence."}],
    max_tokens=256,
)
msg = r.choices[0].message
print(msg.content)
print(getattr(msg, "reasoning_content", None))   # reasoning models return this separately
```

---

## 7. Optional: Open WebUI (browser chat)

Run the UI as a **separate small CPU-only job** pointed at your model (not co-hosted). Easiest is
`serve_vllm.sh --webui` (it finds the model's node and points the UI there). Standalone:

```bash
./serve_openwebui.sh --model-url http://<model-node>.int.janelia.org:8000/v1
# waits for the UI job to start, then prints the URL + SSH tunnel, e.g.:
#   ssh -L 8080:<ui-node>.int.janelia.org:8080 login.int.janelia.org   → http://localhost:8080
```

- **No image to build** (`ghcr.io/open-webui/open-webui:main`), runs via rootless Podman on the
  `local` queue, `--network=host`, `WEBUI_AUTH=False` (fine only behind the SSH tunnel — do not
  expose publicly).
- **Requires tool calling enabled on the model** (§5), or chat returns the `tool_choice` error.

---

## 8. Reaching it from your machine

Servers bind `0.0.0.0`, but compute nodes usually aren't directly browsable — tunnel via a host
that can route to them:
```bash
ssh -L 8000:<node>.int.janelia.org:8000 login.int.janelia.org   # model API at localhost:8000/v1
ssh -L 8080:<node>.int.janelia.org:8080 login.int.janelia.org   # Open WebUI at localhost:8080
```

---

## 9. Tuning

- **`--max-model-len`** — 32768 default; raise as KV-cache budget allows (weights first, the rest
  of HBM becomes KV cache). Bigger context ⇒ fewer concurrent sequences.
- **`--gpu-mem-util`** — 0.90 default; raise cautiously for more KV cache.
- **`--tp`** — set to the number of GPUs for a model that doesn't fit on one.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| bsub **hangs** on submit | Requested CPUs exceed the queue's slots/GPU ratio — vLLM stalls on a warning. Match the ratio (12/GPU, 8 on L4); `serve_vllm.sh` does this automatically. |
| `OSError: Read-only file system: '/nrs'` at import (apptainer) | Host env leaked in. Use `apptainer --cleanenv` and point caches at `/scratch` (the script does). |
| `Read-only file system: '.../flashinfer_cubin/...'` (MXFP4 MoE, e.g. gpt-oss) | FlashInfer JITs cubins into the read-only image. Set `FLASHINFER_CUBIN_DIR` to a writable `/scratch` dir (the script does). |
| `"auto" tool choice requires …` on chat | Serve with `--enable-auto-tool-choice --tool-parser … --reasoning-parser …` (§5). |
| Job killed at ~2 h | No `-W` → 120-min queue default. Pass a large `--walltime`. |
| Podman `Found 0 CDI devices` | CDI not generated on the node — admin runs `nvidia-ctk cdi generate` (§3c). |
| Podman re-pulls the image every job | Its store is node-local `/scratch` (purged). Prefer Apptainer/SIF — no pull ever. |
| `"python": not found` in a container | It's `python3`, not `python`. |
| `CUDA error 802 / no GPUs found` (B300) | Node-level: NVIDIA Fabric Manager not running. |
| `mount source /misc/hf/... doesn't exist` on some nodes | That node group's automount map has **no key** for `/misc/hf` (or `/misc/local`), so the path can't mount there. An admin adds the map key; once present it mounts on access normally (no pre-mount needed). |
| Health check / client hits the **wrong model** on a shared node | Two servers on one node both bound `:8000`. Use a distinct `--port` per server when co-locating on a node. |

---

## 11. Model catalog — what each is good for

Brief, practical notes (all served identically via `serve_vllm.sh`):

- **`openai/gpt-oss-120b` / `gpt-oss-20b`** — OpenAI open-weight MoE (MXFP4), strong general
  reasoning, agentic/tool use, and coding. 120b is the frontier-quality option (2× H200 / 1× B300);
  20b is fast and cheap (1 GPU) for iteration, drafting, and high-throughput tasks.
- **`Qwen/Qwen3-32B`** — dense 32B, excellent multilingual, coding, and instruction following;
  a strong general-purpose default when you want dense-model quality on ~1 big GPU.
- **`Qwen/Qwen3-30B-A3B`** — MoE with only ~3B active params: **very fast / cheap** per token while
  near 30B quality. Great default for interactive chat and high request volume.
- **`Qwen/Qwen2.5-72B-Instruct`** — dense 72B, top-tier general/coding/multilingual quality when you
  want the strongest dense model; needs 2× H200 (TP=2) / B300. Instruct (not a reasoning model). Also
  the base model for the FSDP full fine-tune in [`../finetune/`](../finetune/).
- **`google/gemma-4-26B-A4B-it`** — Google's efficient instruction-tuned MoE; solid general
  assistant, summarization, and safe/aligned responses; fits one big GPU.
- **`mistralai/Mistral-Small-4-119B-2603`** — Mistral's efficient large model; strong general +
  coding + European-language performance; needs 2× H200 / B300.
- **`moonshotai/Kimi-K3`** — frontier agentic model: long-horizon coding, native multimodality
  (text+image), 1M-token context. Whole B300 node — see **[KIMI_K3_VLLM_GUIDE.md](KIMI_K3_VLLM_GUIDE.md)**.

> Version names/dates evolve; treat strengths as typical for the family. Benchmark on your own
> tasks before committing to one for production.

---

*`serve_vllm.sh` and `serve_openwebui.sh` live alongside this guide. Model paths under `/misc/hf`
and the shared SIFs under `/misc/local/singularity` are cluster-specific; everything else is generic.*
