1
import argparse
import itertools
from itertools import product, combinations
import sys
import subprocess
import os
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="comms.py driver")
    parser.add_argument("--par", action="store_true", default=False)
    parser.add_argument("--backend-gloo", action="store_true", default=False)

    args = parser.parse_args()

    comms_extension = 'par' if args.par else 'py'
    driver_abs_path = os.getcwd() + os.path.dirname(__file__)
    comms_abs_path = str(Path(driver_abs_path).parents[1]) + f'/param/train/comms/pt/comms.{comms_extension}'

    backend_type = '--backend gloo' if args.backend_gloo else ''

    master_ip_l = ['localhost']
    num_processes_l = [1]
    processes_per_node_l = [8]
    num_compute_iters_l = [100]
    mm_dim_l = [1000]
    collective_config_l = itertools.chain(
        list(product(["all_reduce"],[2200, 4000, 9944, 22374, 16000, 32000])),
        list(product(["all_to_all"],[134000000, 244000000, 544000000])))



    cli_args = sys.argv
    for (master_ip, num_processes, processes_per_node, num_compute_iters, mm_dim, collective_config) in product(
        master_ip_l, num_processes_l, processes_per_node_l, num_compute_iters_l, mm_dim_l, collective_config_l):

        cmd = f'''
            mpirun
            -np 1
            -N 8
            --bind-to none {comms_abs_path}
            --b 256
            --e 256M
            --f 2
            --n 100
            --master-ip {master_ip}
            --collective {collective_config[0]}
            --b {collective_config[1]}
            --e {collective_config[1]}
            --num-compute {num_compute_iters}
            --mm-dim {mm_dim}
            --master-port 22565
            {backend_type}
        '''

        cmd = cmd.replace('\n', " ")
        cmd = cmd.replace('  ', "")
        print("\n")
        print(cmd)
        print("\n")
        cmd = cmd.split()
        proc = subprocess.run(cmd)

if __name__ == "__main__":
    main()
