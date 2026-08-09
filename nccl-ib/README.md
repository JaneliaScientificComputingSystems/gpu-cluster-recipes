# nccl-ib — shared NCCL / InfiniBand environment configs

These files hold the NCCL InfiniBand settings for the cluster's GPU nodes. They are **sourced by
the launchers** in [`../ray/`](../ray/) (`submit_ray_job.sh`) and [`../torchrun/`](../torchrun/)
(`submit_torchrun_job.sh`) on InfiniBand queues — you don't run them directly.

| File | For |
|---|---|
| `nccl_ib.sh` | H100 / H200 nodes — HCAs `mlx5_0…8`, GDR, QPs, timeouts, `NCCL_SOCKET_IFNAME` |
| `nccl_ib_b300.sh` | B300 nodes (XDR 800 Gb/s) — HCAs `mlx5_0,1,7,8,9,10,14,15`, `NCCL_SOCKET_IFNAME=eno16995np0` |

Both are sourced automatically based on the LSF queue (e.g. `gpu_b300*` → `nccl_ib_b300.sh`,
other IB queues → `nccl_ib.sh`); Ethernet queues (L4/A100) skip these and use socket NCCL.

> If NCCL falls back to `NET/Socket` on an IB queue, the HCA names likely changed — check
> `ibv_devinfo` on the node and update `NCCL_IB_HCA` here.
