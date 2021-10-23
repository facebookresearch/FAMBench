import argparse
import itertools
import os
import pathlib
import subprocess
import sys
from itertools import product


def main():
    parser = argparse.ArgumentParser(description="comms.py driver")
    parser.add_argument(
        "--backend",
        type=str,
        default=("nccl"),
        choices=["nccl", "gloo", "mpi", "ucc", "xla"],
    )
    args = parser.parse_args()

    comms_abs_path = str(
        pathlib.Path(__file__).parents[2].resolve() / "param/train/comms/pt/comms.py"
    )

    master_ip_l = ["localhost"]
    num_processes_l = [1]
    processes_per_node_l = [8]
    num_compute_iters_l = [100]
    mm_dim_l = [1000]
    collective_config_l = itertools.chain(
        product(["all_reduce"], [2200, 4000, 9944, 22374, 16000, 32000]),
        product(["all_to_all"], [134000000, 244000000, 544000000]),
    )

    for (
        master_ip,
        num_processes,
        processes_per_node,
        num_compute_iters,
        mm_dim,
        collective_config,
    ) in product(
        master_ip_l,
        num_processes_l,
        processes_per_node_l,
        num_compute_iters_l,
        mm_dim_l,
        collective_config_l,
    ):
        cmd = f"""
            mpirun
            -np 1
            -N 8
            --bind-to none {comms_abs_path}
            --f 2
            --n 100
            --master-ip {master_ip}
            --master-port 22565
            --collective {collective_config[0]}
            --b {collective_config[1]}
            --e {collective_config[1]}
            --num-compute {num_compute_iters}
            --mm-dim {mm_dim}
            --backend {args.backend}
        """

        cmd = cmd.replace("\n", " ")
        cmd = cmd.replace("  ", "")
        print("\n")
        print(cmd)
        print("\n")
        cmd = cmd.split()
        proc = subprocess.run(cmd)


if __name__ == "__main__":
    main()
