#!/usr/bin/env python3
"""
Persistent rank-0 metric logging for Ray Train jobs.

Ray Train forwards worker stdout to the driver in batches; the final burst is
dropped when workers are torn down on `ray stop`, so per-iter/per-epoch progress
and the final throughput line never reach the job's .out file. This writes those
lines straight to a file from the rank-0 worker (line-buffered + fsync), so they
survive regardless of Ray's stdout forwarding.

Usage (driver, in main()):
    from train_logging import make_log_path
    log_path = make_log_path("gpt2_2b_fsdp")          # absolute path under ../logs
    train_loop_config = {..., "log_path": log_path}

Usage (worker, in train_func):
    from train_logging import MetricLogger
    logger = MetricLogger(config.get("log_path"), world_rank)
    logger.log(f"iter {it} | loss {loss:.4f}")        # prints AND persists on rank 0
    ...
    logger.close()
"""
import datetime
import os


def make_log_path(run_name, log_dir=None):
    """Build an absolute, timestamped log path. Call from the driver so all
    workers receive the same path. Defaults to <cwd>/../logs."""
    if log_dir is None:
        log_dir = os.path.join(os.path.abspath(os.getcwd()), "..", "logs")
    log_dir = os.path.abspath(log_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{run_name}_{ts}.log")


class MetricLogger:
    """Rank-0-only logger: prints (so it still shows in .out when forwarding
    works) and appends to a file with immediate flush+fsync (so it survives
    teardown). A no-op on all other ranks."""

    def __init__(self, log_path, world_rank):
        self.world_rank = world_rank
        self.fh = None
        if world_rank == 0 and log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # buffering=1 -> line-buffered text mode
            self.fh = open(log_path, "a", buffering=1)
            self.log_path = log_path

    def log(self, msg):
        if self.world_rank != 0:
            return
        print(msg, flush=True)
        if self.fh is not None:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.fh.write(f"{ts}  {msg}\n")
            self.fh.flush()
            os.fsync(self.fh.fileno())

    def close(self):
        if self.fh is not None:
            self.fh.flush()
            os.fsync(self.fh.fileno())
            self.fh.close()
            self.fh = None
