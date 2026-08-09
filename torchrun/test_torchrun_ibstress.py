"""
Multi-node IB bandwidth stress test for NCCL (torchrun) — a PyTorch stand-in for
nccl-tests all_reduce_perf, launchable via submit_torchrun_job.sh.

Two phases:
  1. Size sweep  — doubles message size and reports algbw/busbw (GB/s) per size,
                   the standard NCCL bandwidth curve.
  2. Sustained   — hammers the largest size back-to-back (NO sleep) for --secs,
                   printing rolling busbw. This is the painful part: it keeps the
                   inter-node fabric saturated so a flaky link/QP shows up.

With >1 node NCCL uses NVLink intra-node + IB inter-node, so large messages make
IB the bottleneck; busbw is what you compare to the ~cluster IB line rate.

busbw = algbw * 2*(N-1)/N  (N = world size), matching nccl-tests conventions.

Launch (4 nodes / 32 GPUs, ~10 min hammer at 4 GB messages):
    ./submit_torchrun_job.sh 4 --queue=gpu_b300_parallel --venv=~/ray_env \
        --job-name=ibstress --walltime=0:30 \
        --script=~/Projects/gpu-cluster-recipes/torchrun/test_torchrun_ibstress.py -- \
        --sweep-min-mb 8 --max-mb 4096 --secs 600

Args (passed after `--`):
    --sweep-min-mb  smallest message in the sweep, MB   (default 8)
    --max-mb        largest message; also the sustained size, MB (default 2048)
    --sweep-iters   timed iters per sweep size           (default 20)
    --secs          sustained-hammer seconds at --max-mb (default 300; 0 = skip)
    --dtype         float32 | bfloat16 | float16         (default float32)
"""

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def bench(tensor, iters, warmup, world):
    """Time `iters` back-to-back all_reduces; return (sec/iter, algbw, busbw) in GB/s."""
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    nbytes = tensor.numel() * tensor.element_size()
    algbw = nbytes / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world
    return dt, algbw, busbw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-min-mb", type=float, default=8)
    p.add_argument("--max-mb", type=float, default=2048)
    p.add_argument("--sweep-iters", type=int, default=20)
    p.add_argument("--secs", type=float, default=300)
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    args = p.parse_args()
    dtype = _DTYPES[args.dtype]

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.cuda.current_device()
    is0 = rank == 0
    esize = torch.tensor([], dtype=dtype).element_size()

    print(f"[Rank {rank}/{world}] host={socket.gethostname()} local_rank={local} "
          f"gpu={torch.cuda.get_device_name(dev)}", flush=True)
    if is0:
        print(f"[ibstress] world={world} dtype={args.dtype} "
              f"busbw_factor={2 * (world - 1) / world:.3f}", flush=True)

    # ---- Phase 1: size sweep (bandwidth curve) ----
    if is0:
        print(f"\n{'size':>10} {'iters':>6} {'time_ms':>10} {'algbw_GB/s':>12} {'busbw_GB/s':>12}",
              flush=True)
    mb = args.sweep_min_mb
    while mb <= args.max_mb:
        numel = int(mb * 1024 * 1024 / esize)
        t = torch.ones(numel, dtype=dtype, device=dev)
        dt, algbw, busbw = bench(t, args.sweep_iters, 5, world)
        if is0:
            print(f"{mb:>8.0f}MB {args.sweep_iters:>6} {dt * 1e3:>10.2f} "
                  f"{algbw:>12.2f} {busbw:>12.2f}", flush=True)
        del t
        torch.cuda.empty_cache()
        mb *= 2

    # ---- Phase 2: sustained hammer at max size ----
    if args.secs > 0:
        numel = int(args.max_mb * 1024 * 1024 / esize)
        t = torch.ones(numel, dtype=dtype, device=dev)
        nbytes = t.numel() * t.element_size()
        if is0:
            print(f"\n[ibstress] sustained hammer: {args.max_mb:.0f}MB all_reduce for "
                  f"{args.secs:.0f}s, no sleep", flush=True)
        for _ in range(5):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        it = last_it = 0
        last = start
        while time.perf_counter() - start < args.secs:
            dist.all_reduce(t)
            it += 1
            if it % 20 == 0:
                torch.cuda.synchronize()
                now = time.perf_counter()
                if is0 and now - last >= 5:
                    d = (now - last) / (it - last_it)
                    algbw = nbytes / d / 1e9
                    busbw = algbw * 2 * (world - 1) / world
                    print(f"[ibstress] t={now - start:6.1f}s  it={it:6d}  "
                          f"busbw={busbw:7.2f} GB/s", flush=True)
                    last, last_it = now, it
        torch.cuda.synchronize()
        total = time.perf_counter() - start
        if is0:
            algbw = nbytes * it / total / 1e9
            busbw = algbw * 2 * (world - 1) / world
            print(f"\n[PASS] sustained {it} all_reduces of {args.max_mb:.0f}MB over "
                  f"{total:.1f}s on {world} GPUs — avg busbw {busbw:.2f} GB/s, no hang",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
