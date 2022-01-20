# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import pathlib
from os import fspath
# param ubenches
p = pathlib.Path(__file__).parent.resolve() / "../../../param/inference/compute/pt"
sys.path.append(fspath(p))
import pytorch_linear as klinear

# FB5 Logger
p = pathlib.Path(__file__).parent.resolve() / "../../../bmlogging"
sys.path.append(fspath(p))
from bmlogger import get_bmlogger
import loggerconstants

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measuring the Inference Compute Kernel Performance Using PyTorch"
    )
    parser.add_argument('--warmups', type=int, default=10, help="warmup times")
    parser.add_argument('--steps', type=int, default=100, help="repeat times")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'tpu'], required=True, help='valid devices')
    parser.add_argument("--fb5logger", type=str, default=None)

    subparsers = parser.add_subparsers(title='kernels', dest='kernel')
    subparsers.required = True

    parser_linear = subparsers.add_parser('linear', help='measure mlp performance')
    parser_linear.add_argument('--optimizer-type', default='sgd', help='Optimizer: SGD', choices=['sgd'])
    parser_linear.add_argument('-t', '--dtype', default='float', help="data type", choices=["float", "float16", "bfloat16"])
    parser_linear.add_argument('-d', '--dataset', default='small')
    parser_linear.add_argument('--use-trt', default=False, action='store_true')
    parser_linear.add_argument('--use-migraphx', default=False, action='store_true')

    # BMLogging

    args=parser.parse_args()

    print("Measuring the performance of ", args.kernel, " on device = ", args.device)
    print("Steps = ", args.steps, " warmups = ", args.warmups)

    #BM logging header
    if args.fb5logger is not None:
        bmlogger = get_bmlogger(log_file_path=args.fb5logger)
    else:
        bmlogger = get_bmlogger(log_file_path=None) # default to Nop logger

    if args.kernel == 'linear':
        print("with linear dataset ", args.dataset, ", Data type: ", args.dtype)
        global_flops = 0
        global_elap = 0
        bmlogger.header("DLRM", "UBENCH", "infer", args.kernel + "_" + args.dataset, score_metric=loggerconstants.TFPS)
        bmlogger.run_start()
        if args.dataset == 'small':
            small_dataset = [ (18, 1024, 1024, 1024, 128),
                        (18, 1024, 1024, 1024, 256),]
            run_dataset = small_dataset
        else:
            import ast
            run_dataset = ast.literal_eval(args.dataset)
        for i in range(len(run_dataset)):
            layer_num, input_size, hidden_size, output_size, batch_size = run_dataset[i]
            elap = klinear.run_single(
                args, layer_num, input_size, hidden_size, output_size, batch_size
            )
            elap /= args.steps

            flops = batch_size * (
                hidden_size * hidden_size * layer_num
                + hidden_size * input_size
                + hidden_size * output_size
            )
            # Forward 2x
            flops *= 2
            global_flops += flops
            global_elap += elap
        extra_metadata={"TF/s": global_flops / global_elap / 1.0e12, "ELAP": global_elap, "FLOPS": global_flops}
        bmlogger.run_stop(args.steps, batch_size, extra_metadata=extra_metadata)
