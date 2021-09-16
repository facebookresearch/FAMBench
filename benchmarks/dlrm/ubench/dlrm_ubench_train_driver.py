# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import pathlib
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

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measuring the Compute Kernel Performance Using PyTorch"
    )
    parser.add_argument('--warmups', type=int, default=10, help="warmup times")
    parser.add_argument('--steps', type=int, default=100, help="repeat times")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'tpu'], required=True, help='valid devices')

    subparsers = parser.add_subparsers(title='kernels', dest='kernel')
    subparsers.required = True

    parser_gemm = subparsers.add_parser('gemm', help='measure mm performance (m,k)*(k,n)=(m,n)')
    parser_gemm.add_argument('-t', '--dtype', type=str, default="float32")
    parser_gemm.add_argument('-d', '--dataset', choices=['A', 'B', 'C'], default='A')

    parser_emb = subparsers.add_parser('emb', help='measure EmbeddingBag performance')
    parser_emb.add_argument('-d', '--dataset', choices=['A', 'B'], default='A')
    parser_emb.add_argument("--randomseed", type=int, default=0)
    parser_emb.add_argument("--usexlabag", action='store_true', help='use xlabad instead of embeddingbag')
    parser_emb.add_argument("--alpha", default=0.0, help="Zipf param. Use uniform if == 0.0")

    parser_linear = subparsers.add_parser('linear', help='measure mlp performance')
    parser_linear.add_argument('--optimizer-type', default='sgd', help='Optimizer: SGD', choices=['sgd'])
    parser_linear.add_argument('-t', '--dtype', default='float', help="data type", choices=["float", "float16", "bfloat16"])
    parser_linear.add_argument('-d', '--dataset', choices=['A'], default='A')

    # FB5 Logging
    parser.add_argument("--fb5logger", type=str, default=None)

    args=parser.parse_args()

    print("Measuring the performance of ", args.kernel, " on device = ", args.device)
    print("Steps = ", args.steps, " warmups = ", args.warmups)

    #fb5 logging header
    if args.fb5logger is not None:
        fb5logger = FB5Logger(args.fb5logger)
        fb5logger.header("DLRM", "UBENCH", "train", args.kernel)

    if args.kernel == 'emb':
        print("with emb dataset ", args.dataset)
        if args.dataset == 'A':
            kemb.run(args, dataset.emb_A)
        elif args.dataset == 'B':
            kemb.run(args, dataset.emb_B)

    else:
        print("with linear dataset ", args.dataset, ", Data type: ", args.dtype)
        total_flops = 0
        total_elap = 0
        if args.dataset == 'A':
            fb5logger.run_start()
            for i in range(len(dataset.mlp_A)):
                layer_num, input_size, hidden_size, output_size, batch_size = dataset[i]
                elap, loss = klinear.run_single(
                    args, layer_num, input_size, hidden_size, output_size, batch_size
                )
                elap /= args.steps

                flops = batch_size * (
                    hidden_size * hidden_size * layer_num
                    + hidden_size * input_size
                    + hidden_size * output_size
                )
                # Forward 2x and Backward 4x
                flops *= 6
                total_flops += flops
                total_elap += elap
            extra_metadata={"TF/s": total_flops / total_elap / 1.0e12, "ELAP": total_elap, "FLOPS": total_flops}
            fb5logger.run_stop(args.steps, 1, extra_metadata=extra_metadata)
