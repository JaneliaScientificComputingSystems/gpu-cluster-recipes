#!/usr/bin/env python3
"""
Small helpers shared by the Ray Train recipes.

global_sum: sum per-rank counters across all ranks, so a metric is computed
from global totals (correct / total) instead of averaging per-rank ratios.
Averaging ratios is only right when every rank saw the same number of samples,
and it reports rank 0's share of the data when not reduced at all.
"""
import torch
import torch.distributed as dist


def global_sum(values, device):
    """Return the element-wise sum of `values` over all ranks, as floats.
    Every rank must call this the same number of times. A no-op without an
    initialised process group or with a single worker."""
    values = [float(v) for v in values]
    if not (dist.is_available() and dist.is_initialized()):
        return values
    if dist.get_world_size() == 1:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()
