# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import pathlib
from os import fspath
# embeddingbag fbgemm based ubench
import torch
import numpy as np
p = pathlib.Path(__file__).parent.resolve() / "../../../FBGEMM/fbgemm_gpu"
sys.path.append(fspath(p))
import bench.split_table_batched_embeddings_benchmark

from fbgemm_gpu.split_table_batched_embeddings_ops import (
    BoundsCheckMode,
    CacheAlgorithm,
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
    OptimType,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
# FB5 Logger
p = pathlib.Path(__file__).parent.resolve() / "../../../bmlogging"
sys.path.append(fspath(p))
from bmlogger import get_bmlogger
import loggerconstants
import statistics
import time

PRECISION_SIZE_MULTIPLIER = {
    SparseType.FP32: 4,
    SparseType.FP16: 2,
    SparseType.INT8: 1,
    SparseType.INT4: 0.5,
}
def str_to_sparsetype(dtype):
    str_to_sparsetype_dict = {
            "float": SparseType("fp32"),
            "float16": SparseType("fp16"),
            "int8": SparseType("int8"),
            "int4": SparseType("int4"),
    }
    return str_to_sparsetype_dict[dtype]


def run_emb(args, run_dataset):
    # Assumption is that all tablesare identical in terms of shape, number of accesses and batch size
    assert(len(run_dataset) == 1)
    B = run_dataset[0][3]
    T = run_dataset[0][4]
    Ds = [run_dataset[0][1]] * T
    D = np.average(Ds)
    E = run_dataset[0][0]
    L = run_dataset[0][2]
    weights_precision = str_to_sparsetype(args.weights_precision)
    output_dtype = str_to_sparsetype(args.output_dtype)

    forward_only = args.forward_only

    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD if args.row_wise else OptimType.EXACT_ADAGRAD
    managed_option = (
        EmbeddingLocation.DEVICE
        if torch.cuda.is_available()
        else EmbeddingLocation.HOST
    )

    if weights_precision == SparseType.INT4 or weights_precision == SparseType.INT8:
        # this is inference only, so no optimzer
        emb = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, d, weights_precision, managed_option) for d in Ds],
            bounds_check_mode=BoundsCheckMode.WARNING,
            output_dtype=output_dtype,
        ).cuda()
        emb.initialize_weights()
        forward_only = True
    else:
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [(E, d, managed_option,
                    ComputeDevice.CUDA
                    if torch.cuda.is_available()
                    else ComputeDevice.CPU,
                )
                for d in Ds
            ],
            optimizer=optimizer,
            learning_rate=0.1,
            eps=0.1,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        ).cuda()
    isIntNTableBatched = isinstance(emb, IntNBitTableBatchedEmbeddingBagsCodegen)

    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]

    print(
        f"Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {args.weighted}, "
    )
    requests = bench.split_table_batched_embeddings_benchmark.generate_requests(
        args.warmups+args.steps,
        B,
        T,
        L,
        E,
        alpha=args.alpha,
        weights_precision=args.weights_precision,
        weighted=args.weighted,
    )
    if isIntNTableBatched:
        requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]
    warmup_requests, requests = requests[:args.warmups], requests[args.warmups:]

    #warmups
    for (indices, offsets, weights) in warmup_requests:
        emb.forward(indices, offsets, weights)

    # forward
    time_per_iter = bench.split_table_batched_embeddings_benchmark.benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb.forward(
            indices,
            offsets,
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=args.flush_gpu_cache_size_mb,
    )
    bytes_per_iter = B * L * D * T * param_size_multiplier

    if forward_only:
        return time_per_iter, bytes_per_iter

    grad_output = torch.randn(B, sum(Ds)).cuda()
    # backward
    time_per_iter = bench.split_table_batched_embeddings_benchmark.benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb(
            indices.long(),
            offsets.long(),
            per_sample_weights,
        ).backward(grad_output),
        flush_gpu_cache_size_mb=args.flush_gpu_cache_size_mb,
    )
    bytes_per_iter = B * L * D * T * param_size_multiplier * 3

    return time_per_iter, bytes_per_iter

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measuring the EmbeddingBag Kernel Performance Using PyTorch"
    )
    parser.add_argument('--warmups', type=int, default=10, help="warmup times")
    parser.add_argument('--steps', type=int, default=100, help="repeat times")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'tpu'], required=True, help='valid devices')
    parser.add_argument("--fb5logger", type=str, default=None)
    parser.add_argument('-d', '--dataset', default='B')
    parser.add_argument("--randomseed", type=int, default=0)
    parser.add_argument("--alpha", default=0.0, help="Zipf param. Use uniform if == 0.0")
    parser.add_argument("--reuse", default=0.0, help="Amount of reuse. 0.0 indicates no reuse.")
    parser.add_argument('-t', '--weights_precision', default='float', help="data type", choices=["float",
        "float16", "int8", "int4"])
    parser.add_argument('--row-wise', dest='row_wise', action='store_true')
    parser.add_argument('--no-row-wise', dest='row_wise', action='store_false')
    parser.set_defaults(row_wise=True)
    parser.add_argument('--weighted', dest='weighted', action='store_true')
    parser.set_defaults(weighted=False)
    parser.add_argument('--flush-gpu-cache-size-mb', type=int, default=0, help="flush gpu cache size")
    parser.add_argument('--dense', dest='dense', action='store_true')
    parser.set_defaults(dense=False)
    parser.add_argument('--output_dtype', default='float', help="data type", choices=["float", "float16"])
    parser.add_argument('--forward_only', dest='forward_only', action='store_true')
    parser.set_defaults(forward_only=False)

    # BM Logging

    args=parser.parse_args()

    print("Measuring the performance of EmbeddingBag on device = ", args.device)
    print("Steps = ", args.steps, " warmups = ", args.warmups)

    #bmlogging header
    if args.fb5logger is not None:
        bmlogger = get_bmlogger(log_file_path=args.fb5logger)
    else:
        bmlogger = get_bmlogger(log_file_path=None) # default to Nop logger

    print("with emb dataset ", args.dataset)
    global_bytes = 0
    global_elap = 0
    if args.fb5logger is not None:
        bmlogger.header("DLRM", "UBENCH", "train", "emb_" + args.dataset, score_metric=loggerconstants.GBPS)
        bmlogger.run_start()
    if args.dataset == 'small':
        small_dataset = [ (4800000, 56, 34, 2048, 2) ]
        run_dataset = small_dataset
    else:
        import ast
        run_dataset = ast.literal_eval(args.dataset)

    global_elap, global_bytes = run_emb(args, run_dataset)

    if args.fb5logger is not None:
        extra_metadata={"GB/s": global_bytes / global_elap / 1.0e9, "ELAP": global_elap, "BYTES": global_bytes}
        bmlogger.run_stop(args.steps, run_dataset[0][3], extra_metadata=extra_metadata)
