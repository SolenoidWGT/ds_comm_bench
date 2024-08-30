"""Copyright The Microsoft DeepSpeed Team"""

import sys, os


from utils import *
from all_reduce import run_all_reduce
from reduce import run_reduce
from all_gather import run_all_gather
from all_to_all import run_all_to_all
from pt2pt import run_pt2pt
from reduce_sactter import run_reduce_sactter
from broadcast import run_broadcast
from constants import *


# For importing
def main(args, rank):

    init_processes(local_rank=rank, args=args)

    ops_to_run = []
    if args.all_reduce:
        ops_to_run.append("all_reduce")
    if args.all_gather:
        ops_to_run.append("all_gather")
    if args.broadcast:
        ops_to_run.append("broadcast")
    # if args.pt2pt:
    #     ops_to_run.append("pt2pt")
    # if args.all_to_all:
    #     ops_to_run.append("all_to_all")

    if len(ops_to_run) == 0:
        ops_to_run = ["all_reduce", "all_gather", "broadcast", "reduce_sactter", "reduce"]

    for comm_op in ops_to_run:
        if comm_op == "all_reduce":
            run_all_reduce(local_rank=rank, args=args)
        if comm_op == "all_gather":
            run_all_gather(local_rank=rank, args=args)
        if comm_op == "all_to_all":
            run_all_to_all(local_rank=rank, args=args)
        if comm_op == "pt2pt":
            run_pt2pt(local_rank=rank, args=args)
        if comm_op == "broadcast":
            run_broadcast(local_rank=rank, args=args)
        if comm_op == "reduce_sactter":
            run_reduce_sactter(local_rank=rank, args=args)
        if comm_op == "reduce":
            run_reduce(local_rank=rank, args=args)


# For directly calling benchmark
if __name__ == "__main__":
    # os.environ["NCCL_IB_HCA"] = "mlx5_2,mlx5_3,mlx5_4,mlx5_5"
    rank = int(os.environ['RANK'])
    local_rank = rank % 8
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)

    args = benchmark_parser().parse_args()
    args.local_rank = local_rank    

    main(args, rank)
