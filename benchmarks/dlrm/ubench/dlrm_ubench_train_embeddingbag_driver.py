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

def generate_requests(
        iters,
        run_dataset,
        alpha,
        weights_precision,
        weighted,
):
    rs = []

    for it in range(iters):
        print('Generating data for iter ', it, ' of ', iters, end='\r')
        indices = []
        lengths = []

        for t in run_dataset:
            E = t[0]
            B = t[3]
            L = t[2]
            D = t[1]
            if alpha <= 1.0:
                local_indices = torch.randint(
                   low=0,
                   high=E,
                   size=(B, L),
                   dtype=torch.int32,
                )
                (local_indices, _) = torch.sort(local_indices)
            else:
                local_indices = (np.random.zipf(a=alpha, size=(B , L * 3)) - 1) % E
                for b in range(B):
                    r = set()
                    for x in local_indices[b]:
                        if x not in r:
                            r.add(x)
                            if len(r) == L:
                                break
                    assert (len(r)) == L, "too skewed distribution (alpha too big)"
                    local_indices[b][:L] = list(r)
                # shuffle indices so we don't have unintended spatial locality
                local_indices = torch.as_tensor(local_indices[:, :L])
                rng = default_rng()
                permutation = torch.as_tensor(
                    rng.choice(E, size=local_indices.max().item() + 1, replace=False)
                )
                local_indices = permutation.gather(0, local_indices.flatten())
            indices.append(local_indices.reshape(B * L))
            lengths.extend([L] * B)
        indices = torch.cat(indices)
        assert indices.shape[0] == sum(t[3] * t[2] for t in run_dataset)
        weights_tensor = (
            None if not weighted else torch.randn(T * B * L, device='cuda')
        )
        rs.append(
            (
                indices.long().contiguous().cuda(),
                torch.tensor(([0] + np.cumsum(lengths).tolist())).long().cuda(),
            )
            + (weights_tensor,)
        )
    return rs


def benchmark_requests(
    requests,
    func,
    flush_gpu_cache_size_mb = 0,
    check_median = False,
) -> float:
    times = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    for (indices, offsets, weights) in requests:
        start_time = time.time()
        if torch.cuda.is_available():
            if flush_gpu_cache_size_mb:
                _ = torch.rand(
                    flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
                )
                torch.cuda.synchronize()
            start_event.record()
        func(indices, offsets, weights)
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            it_time = start_event.elapsed_time(end_event) * 1.0e-3
            times.append(it_time)
        else:
            it_time = time.time() - start_time
            times.append(it_time)
    avg_time = sum(times) / len(requests)
    median_time = statistics.median(times)
    return median_time if check_median else avg_time


def run_emb(args, run_dataset):
    # Assumption is that batch size is the same for all tables
    B = run_dataset[0][3]
    T = len(run_dataset)
    Ds = [t[1] for t in run_dataset]
    D = np.average(Ds)
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
            [("", t[0], t[1], weights_precision, managed_option) for t in run_dataset],
            bounds_check_mode=BoundsCheckMode.WARNING,
            output_dtype=output_dtype,
        ).cuda()
        emb.initialize_weights()
        forward_only = True
    else:
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [(t[0], t[1], managed_option,
                    ComputeDevice.CUDA
                    if torch.cuda.is_available()
                    else ComputeDevice.CPU,
                )
                for t in run_dataset
            ],
            optimizer=optimizer,
            learning_rate=0.1,
            eps=0.1,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        ).cuda()
    isIntNTableBatched = isinstance(emb, IntNBitTableBatchedEmbeddingBagsCodegen)

    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]


    requests = generate_requests(
        args.warmups+args.steps,
        run_dataset,
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
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb.forward(
            indices,
            offsets,
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=args.flush_gpu_cache_size_mb,
    )
    bytes_per_iter = sum(t[3] * t[2] * t[1] for t in run_dataset) * param_size_multiplier

    if forward_only:
        return time_per_iter, bytes_per_iter

    grad_output = torch.randn(B, sum(Ds)).cuda()
    # backward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb(
            indices.long(),
            offsets.long(),
            per_sample_weights,
        ).backward(grad_output),
        flush_gpu_cache_size_mb=args.flush_gpu_cache_size_mb,
    )
    bytes_per_iter = sum(t[3] * t[2] * t[1] for t in run_dataset) * param_size_multiplier * 3

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
        small_dataset = [ (4800000, 56, 34, 2048),
                    (4800000, 56, 34, 4096),]
        run_dataset = small_dataset
    else:
        import ast
        run_dataset = ast.literal_eval(args.dataset)

    global_elap, global_bytes = run_emb(args, run_dataset)

    if args.fb5logger is not None:
        extra_metadata={"GB/s": global_bytes / global_elap / 1.0e9, "ELAP": global_elap, "BYTES": global_bytes}
        bmlogger.run_stop(args.steps, run_dataset[0][3], extra_metadata=extra_metadata)
