"""
Minimal reproduction of the rank-0 held-out-eval barrier deadlock and its fix.

Mirrors dinov3tcn_ddp.py: ranks run some cross-node NCCL collectives (training
steps), then rank 0 goes off and does a long SOLO computation (the held-out-mouse
fine-tune, up to 500 epochs) while the other ranks must wait. The question is what
the other ranks wait ON:

    MODE=nccl  -> dist.barrier() on the NCCL group. The NCCL watchdog (~480s, here
                 forced low via TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC) declares the
                 waiting ranks "stuck" and aborts the job.  Reproduces the hang.
    MODE=gloo  -> dist.monitored_barrier() on a Gloo group, which is NOT governed
                 by the NCCL watchdog.  Survives the long rank-0 delay.  The fix.

Env:
    MODE        nccl | gloo            (default gloo)
    SLEEP_S     rank-0 solo seconds    (default 540; > the 480s watchdog)
"""

import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    mode = os.environ.get("MODE", "gloo")
    sleep_s = float(os.environ.get("SLEEP_S", 540))

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    host = socket.gethostname()

    print(f"[Rank {rank}/{world_size}] host={host} local_rank={local_rank} MODE={mode} "
          f"sleep_s={sleep_s}", flush=True)

    gloo_group = dist.new_group(backend="gloo") if world_size > 1 else None

    # A few cross-node NCCL collectives to mimic training steps (advances SeqNum).
    t = torch.ones(1024, device=device)
    for _ in range(5):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    if rank == 0:
        print("[repro] warmup collectives done; rank 0 entering long solo work", flush=True)

    # Rank 0 does long SOLO work (the held-out fine-tune). Others must wait.
    if rank == 0:
        start = time.time()
        while time.time() - start < sleep_s:
            time.sleep(5)
            print(f"[repro] rank0 solo work {time.time()-start:.0f}/{sleep_s:.0f}s", flush=True)
        print("[repro] rank0 solo work complete; joining barrier", flush=True)

    # The wait. This is the line under test.
    if world_size > 1:
        if mode == "nccl":
            dist.barrier()
        else:
            dist.monitored_barrier(group=gloo_group,
                                   timeout=timedelta(seconds=7200),
                                   wait_all_ranks=True)

    # Prove the NCCL communicator still works after the long wait.
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"\n[PASS] mode={mode}: all {world_size} ranks cleared the barrier after a "
              f"{sleep_s:.0f}s rank-0 stall, and post-barrier NCCL allreduce works", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
