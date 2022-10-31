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
p = pathlib.Path(__file__).parent.resolve() / "../../../bmlogging"
sys.path.append(fspath(p))
from bmlogger import get_bmlogger
import loggerconstants

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measuring the Compute Kernel Performance Using PyTorch"
    )
    parser.add_argument('--warmups', type=int, default=10, help="warmup times")
    parser.add_argument('--steps', type=int, default=100, help="repeat times")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'tpu'], required=True, help='valid devices')
    parser.add_argument("--fb5logger", type=str, default=None)

    subparsers = parser.add_subparsers(title='kernels', dest='kernel')
    subparsers.required = True

    parser_emb = subparsers.add_parser('emb', help='measure EmbeddingBag performance')
    parser_emb.add_argument('-d', '--dataset', default='B')
    parser_emb.add_argument("--randomseed", type=int, default=0)
    parser_emb.add_argument("--usexlabag", action='store_true', help='use xlabad instead of embeddingbag')
    parser_emb.add_argument("--alpha", default=0.0, help="Zipf param. Use uniform if == 0.0")

    parser_linear = subparsers.add_parser('linear', help='measure mlp performance')
    parser_linear.add_argument('--optimizer-type', default='sgd', help='Optimizer: SGD', choices=['sgd'])
    parser_linear.add_argument('-t', '--dtype', default='float', help="data type", choices=["float", "float16", "bfloat16", "tf32"])
    parser_linear.add_argument('-d', '--dataset', default='small')

    parser_linear = subparsers.add_parser('gemm', help='measure gemm performance')
    parser_linear.add_argument('-t', '--dtype', default='float', help="data type", choices=["float", "float16", "bfloat16", "tf32"])
    parser_linear.add_argument('-d', '--dataset', default='small')
    # FB5 Logging

    args=parser.parse_args()

    print("Measuring the performance of ", args.kernel, " on device = ", args.device)
    print("Steps = ", args.steps, " warmups = ", args.warmups)

    #fb5 logging header
    if args.fb5logger is not None:
        fb5logger = get_bmlogger(args.fb5logger)

    if args.kernel == 'emb':
        print("with emb dataset ", args.dataset)
        global_bytes = 0
        global_elap = 0
        if args.fb5logger is not None:
            fb5logger.header("DLRM", "UBENCH", "train", args.kernel + "_" + args.dataset, score_metric=loggerconstants.GBPS)
            fb5logger.run_start()
        if args.dataset == 'A':
            run_dataset = dataset.emb_A
        elif args.dataset == 'B':
            run_dataset = dataset.emb_B
        elif args.dataset == 'small':
            small_dataset = [ (4800000, 56, 34, 2048),
                        (4800000, 56, 34, 4096),]
            run_dataset = small_dataset
        else:
            import ast
            run_dataset = ast.literal_eval(args.dataset)
            assert(len(run_dataset) == 1)
            run_dataset = [run_dataset[0][:4]] * run_dataset[0][4]
        for i in range(len(run_dataset)):
            features, embdim, nnz, batch = run_dataset[i]
            elap, total_bytes = kemb.run_single(args, features, embdim, nnz, batch)
            elap /= args.steps
            total_bytes /= 1.0e6
            global_bytes += total_bytes
            global_elap += elap
        if args.fb5logger is not None:
            extra_metadata={"GB/s": global_bytes / global_elap / 1.0e3, "ELAP": global_elap, "BYTES": global_bytes}
            fb5logger.run_stop(args.steps, batch, extra_metadata=extra_metadata)
    elif args.kernel == 'linear':
        print("with linear dataset ", args.dataset, ", Data type: ", args.dtype)
        global_flops = 0
        global_elap = 0
        if args.fb5logger is not None:
            fb5logger.header("DLRM", "UBENCH", "train", args.kernel + "_" + args.dataset + "_" + args.dtype, score_metric=loggerconstants.TFPS)
            fb5logger.run_start()
        if args.dataset == 'A':
            run_dataset = dataset.mlp_A
        elif args.dataset == 'small':
            small_dataset = [ (18, 1024, 1024, 1024, 128),
                        (18, 1024, 1024, 1024, 256),]
            run_dataset = small_dataset
        else:
            import ast
            run_dataset = ast.literal_eval(args.dataset)
        for i in range(len(run_dataset)):
            layer_num, input_size, hidden_size, output_size, batch_size = run_dataset[i]
            layers_size = [input_size] + [hidden_size]*layer_num + [output_size]
            layers_size = [int(size) for size in layers_size]
            print(layers_size)
            elap, loss = klinear.run_single(
                args, layers_size, batch_size
            )
            elap /= args.steps

            flops = batch_size * (
                hidden_size * hidden_size * layer_num
                + hidden_size * input_size
                + hidden_size * output_size
            )
            # Forward 2x and Backward 4x
            flops *= 6
            global_flops += flops
            global_elap += elap
        if args.fb5logger is not None:
            extra_metadata={"TF/s": global_flops / global_elap / 1.0e12, "ELAP": global_elap, "FLOPS": global_flops}
            fb5logger.run_stop(args.steps, batch_size, extra_metadata=extra_metadata)
    else:
        print("with gemm dataset ", args.dataset, ", Data type: ", args.dtype)
        global_flops = 0
        global_elap = 0
        if args.fb5logger is not None:
            fb5logger.header("DLRM", "UBENCH", "train", args.kernel + "_" + args.dataset + "_" + args.dtype, score_metric=loggerconstants.TFPS)
            fb5logger.run_start()
        if args.dataset == 'small':
            small_dataset = [ (256, 1024, 1024), ]
            run_dataset = small_dataset
        else:
            import ast
            run_dataset = ast.literal_eval(args.dataset)
        for i in range(len(run_dataset)):
            m, n, k = run_dataset[i]
            elap = kgemm.run_single(args, m, n, k)
            elap /= args.steps

            flops = m * n * k
            # this is pure GEMM
            flops *=2
            global_flops += flops
            global_elap += elap
        if args.fb5logger is not None:
            extra_metadata={"TF/s": global_flops / global_elap / 1.0e12, "ELAP": global_elap, "FLOPS": global_flops}
            fb5logger.run_stop(args.steps, run_dataset[0][0], extra_metadata=extra_metadata)
