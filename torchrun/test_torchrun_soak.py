"""
Sustained NCCL soak test — reproduces mid-run cross-node hangs that the short
test_torchrun.py misses.

The real DINO training hangs happened only AFTER many successful cross-node
allreduces (observed SeqNum 31 and 327), spread over several minutes of training
time. The standard minimal test does only ~13 allreduces in a few seconds, so it
passes even on a link that later stalls. This test loops many small
(DDP-gradient-bucket-sized) allreduces with a short per-iteration sleep so the run
spans minutes like real training, exercising the socket data plane under sustained,
time-extended load.

Each rank logs progress periodically. If any rank stalls, the others trip the
NCCL watchdog (default 480s) exactly as in the training failures.

Tunables via env:
    SOAK_ITERS    total allreduce iterations (default 2000)
    SOAK_NUMEL    elements per allreduce (default 558002 ~ 2.1MB f32, matches a
                  failing job's stuck collective NumelIn)
    SOAK_SLEEP    seconds to sleep between iterations (default 0.3)
    SOAK_LOG_EVERY iterations between progress logs (default 50)

Usage (via submit_torchrun_job.sh):
    ./submit_torchrun_job.sh 2 --queue=gpu_a100_parallel --venv=~/ray_env \
        --job-name=nccl_soak --walltime=0:30 --host="h10u16 h11u20" \
        --script=~/Projects/gpu-cluster-recipes/torchrun/test_torchrun_soak.py
"""

import os
import socket
import time
import torch
import torch.distributed as dist


def main():
    iters = int(os.environ.get("SOAK_ITERS", 2000))
    numel = int(os.environ.get("SOAK_NUMEL", 558002))
    sleep_s = float(os.environ.get("SOAK_SLEEP", 0.3))
    log_every = int(os.environ.get("SOAK_LOG_EVERY", 50))

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    host = socket.gethostname()

    print(f"[Rank {rank}/{world_size}] host={host} local_rank={local_rank} "
          f"gpu={torch.cuda.get_device_name(device)}", flush=True)

    if rank == 0:
        approx_mb = numel * 4 / 1024 / 1024
        print(f"[soak] {iters} allreduces x {numel} elems (~{approx_mb:.1f} MB f32), "
              f"sleep {sleep_s}s/iter, ~{iters * sleep_s / 60:.1f} min total", flush=True)

    tensor = torch.randn(numel, device=device)
    dist.barrier()
    start = time.perf_counter()

    for i in range(1, iters + 1):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        if rank == 0 and (i % log_every == 0 or i == 1):
            elapsed = time.perf_counter() - start
            print(f"[soak] iter {i}/{iters}  elapsed {elapsed:.1f}s  "
                  f"({i / elapsed:.1f} it/s)", flush=True)
        if sleep_s > 0:
            time.sleep(sleep_s)

    dist.barrier()
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    if rank == 0:
        print(f"\n[PASS] soak complete — {iters} cross-node allreduces over "
              f"{total:.1f}s on {world_size} GPUs, no hang", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
