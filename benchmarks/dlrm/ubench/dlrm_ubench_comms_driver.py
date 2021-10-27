import argparse
import itertools
import os
import pathlib
import subprocess
import sys
from itertools import product

from os import fspath
# param ubenches
p = pathlib.Path(__file__).parent.resolve() / "../../../param/train/compute/pt"
sys.path.append(fspath(p))
import dataset
import pytorch_gemm as kgemm
import pytorch_emb as kemb
import pytorch_linear as klinear

# FB5 Logger
p = pathlib.Path(__file__).parent.resolve() / "../../../fb5logging"
sys.path.append(fspath(p))
from fb5logger import FB5Logger
import loggerconstants

def main():
    parser = argparse.ArgumentParser(description="comms.py driver")
    parser.add_argument(
        "--size",
        type=str,
        default="small",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=("nccl"),
        choices=["nccl", "gloo", "mpi", "ucc", "xla"],
    )   
    parser.add_argument(
        "--collective",
        type=str,
        default=("all_to_all"),
        choices=["all_to_all", "all_reduce"],
    )
    parser.add_argument("--fb5logger", type=str, default=None)
    args = parser.parse_args()

    if args.size not in ["small", "medium", "large"] and not (args.size.isdigit() and args.size > 0):
        sys.exit("The --size argument provided is not a valid positive integer.")    

    lookup = {
        "small": 2200 if args.collective=="all_reduce" else 134000000, 
        "medium": 9944 if args.collective=="all_reduce" else 244000000, 
        "large": 22372 if args.collective=="all_reduce" else 544000000,
        str(2200) : "small" if args.collective=="all_reduce" else 2200,
        str(9944) : "medium" if args.collective=="all_reduce" else 9944,
        str(22372) : "large" if args.collective=="all_reduce" else 22372,
        str(134000000) : "small" if args.collective=="all_to_all" else 134000000,
        str(244000000) : "medium" if args.collective=="all_to_all" else 244000000,
        str(544000000) : "large" if args.collective=="all_to_all" else 544000000,
    }
    (x, y) = (args.size, lookup.get(args.size, args.size))
    (size, name) = (x, y) if args.size.isdigit() else (y, x)

    master_ip = "localhost"
    num_compute_per_collective = 100
    mm_dim = 1000
    num_iter = 100

    cmd = f"""
        --f 2
        --n {num_iter}
        --master-ip {master_ip}
        --master-port 22565
        --collective {args.collective}
        --b {size}
        --e {size}
        --num-compute {num_compute_per_collective}
        --mm-dim {mm_dim}
        --backend {args.backend}
    """
    sys.argv = cmd.replace("\n", " ").replace("  ", "").split()

    print("")
    comms_abs_dir_path = str(
        pathlib.Path(__file__).absolute().parents[3].resolve() / "param/train/comms/pt"
    )    
    sys.path.append(comms_abs_dir_path)
    from comms import main as comms_main

    fb5logger = FB5Logger(args.fb5logger)
    fb5logger.header("DLRM", "UBENCH", "train", "comms_" + args.collective.replace("_","") + "_" + name, score_metric=loggerconstants.GBPS

    fb5logger.run_start()

    comms_main()
    
    nbatches = 1
    mini_batch_size = 100
    extra_metadata={"GB/s": global_bytes / global_elap / 1.0e3, "ELAP": global_elap, "BYTES": global_bytes} 
    fb5logger.run_stop(num_iter, num_compute_per_collective, extra_metadata=extra_metadata))

if __name__ == "__main__":
    main()
