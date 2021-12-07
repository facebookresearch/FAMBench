import argparse
import contextlib
import io
import pathlib
import subprocess
import sys
from os import fspath

# param ubenches comms
p = pathlib.Path(__file__).absolute().parents[3].resolve() / "param/train/comms/pt"
sys.path.append(fspath(p))
import comms_utils
from comms import main as comms_main

# FB5 Logger
p = pathlib.Path(__file__).parent.resolve() / "../../../fb5logging"
sys.path.append(fspath(p))
import loggerconstants
from fb5logger import FB5Logger


def get_local_rank():
    mpi_env_params = comms_utils.read_comms_env_vars()
    print(mpi_env_params["local_rank"])


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

    if args.size not in ["small", "medium", "large"] and not (
        args.size.isdigit() and int(args.size) > 0
    ):
        sys.exit("The --size argument provided is not a valid positive integer.")

    lookup = {
        "small": 2200 if args.collective == "all_reduce" else 134000000,
        "medium": 9944 if args.collective == "all_reduce" else 244000000,
        "large": 22372 if args.collective == "all_reduce" else 544000000,
        str(2200): "small" if args.collective == "all_reduce" else 2200,
        str(9944): "medium" if args.collective == "all_reduce" else 9944,
        str(22372): "large" if args.collective == "all_reduce" else 22372,
        str(134000000): "small" if args.collective == "all_to_all" else 134000000,
        str(244000000): "medium" if args.collective == "all_to_all" else 244000000,
        str(544000000): "large" if args.collective == "all_to_all" else 544000000,
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

    fb5logger = FB5Logger(args.fb5logger)
    fb5logger.header(
        "DLRM",
        "UBENCH",
        "train",
        "comms_" + args.collective.replace("_", "") + "_" + name,
        score_metric=loggerconstants.GBPS,
    )

    mpi_env_params = comms_utils.read_comms_env_vars()
    print("This process's MPI global rank: ", mpi_env_params["global_rank"])
    comms_stdout = io.StringIO()
    with contextlib.redirect_stdout(comms_stdout):
        if mpi_env_params["global_rank"] == 0:
            fb5logger.run_start()
        comms_main()

    if mpi_env_params["global_rank"] == 0:
        print(comms_stdout.getvalue())        
        output = comms_stdout.getvalue().split("\n")[-3:]
        output = [" ".join(line.split()).split() for line in output]
        output[0].pop(2)
        output[1].insert(3, "")
        output[0][3] = "Latency(us):"
        output[0].insert(4, "p50")
        extra_metadata = {}
        for a, b in zip(output[0], output[1]):
            extra_metadata[a.lstrip()] = b.lstrip()
        fb5logger.run_stop(
            num_batches=num_iter, batch_size=None, extra_metadata=extra_metadata
        )
        print("-- Pretty Format --")
        for a, b in zip(output[0], output[1]):
            print("{:<18s}{:>4s}".format(a.lstrip(), b.lstrip()))


if __name__ == "__main__":
    main()
