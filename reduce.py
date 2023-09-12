"""Copyright The Microsoft DeepSpeed Team"""

import torch
import sys, os, time


from utils import *
from constants import *
from accelerator import accelerator


def get_accelerator():
    return accelerator


def timed_all_reduce(input, args):
    if args.dist == "torch":
        import torch.distributed as dist
    elif args.dist == "deepspeed":
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        dist.reduce(input, dst=0, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        dist.reduce(input, dst=0, async_op=args.async_op)
    sync_all()
    duration = time.perf_counter() - pre

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw("reduce", size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f"{input.nelement()}x{input.element_size()}x1"

    if not args.raw:
        size = convert_size(size)
    del input
    get_accelerator().empty_cache()

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_reduce(local_rank, args):
    if args.dist == "torch":
        import torch.distributed as dist
    elif args.dist == "deepspeed":
        import deepspeed.comm as dist

    # Prepare benchmark header
    print_header(args, "reduce")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    if args.scan:
        M_LIST = []
        for x in (2**p for p in range(args.minsize, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                mat = torch.ones(world_size, M, dtype=getattr(torch, args.dtype)).to(
                    get_accelerator().device_name(local_rank)
                )
                sync_all()
                input = (mat.mul_(float(global_rank))).view(-1)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if dist.get_rank() == 0:
                        print("WARNING: Ran out of GPU memory. Exiting comm op.")
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_all_reduce(input, args)
            del input
            get_accelerator().empty_cache()
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so we double mem_factor
        elements_per_gpu = max_numel(
            comm_op="reduce",
            dtype=getattr(torch, args.dtype),
            mem_factor=args.mem_factor * 2,
            local_rank=local_rank,
            args=args,
        )
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch, args.dtype)).to(
                get_accelerator().device_name(local_rank)
            )
            input = (mat.mul_(float(global_rank))).view(-1)
        except RuntimeError as e:
            if "out of memory" in str(e):
                if dist.get_rank() == 0:
                    print("WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!")
                sync_all()
                return
            else:
                raise e
        sync_all()
        timed_all_reduce(input, args)


def timed_all_reduce_node_view(input, args):
    if args.dist == "torch":
        import torch.distributed as dist
    elif args.dist == "deepspeed":
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        dist.reduce(input, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        dist.reduce(input, async_op=args.async_op)
    sync_cuda()
    rank_wait_time = time.perf_counter() - pre

    sync_all()
    duration = time.perf_counter() - pre

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw("reduce", size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f"{input.nelement()}x{input.element_size()}"

    if not args.raw:
        size = convert_size(size)

    print_node_0(f"{size:<20} {desc:25s} {duration_str:20s} {rank_wait_time:20s} {tput_str:20s} {busbw_str:20s}")


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_all_reduce(local_rank=rank, args=args)