# Driver to benchmark:
# (1) CvT's convolutional token embedding layers; and
# (2) CvT's convolutional projection layers.

import collections
import pathlib
import sys
from collections import OrderedDict
from itertools import repeat
from os import fspath

# param ubenches
p = pathlib.Path(__file__).parent.resolve() / "../../../param/train/compute/pt"
sys.path.append(fspath(p))
import pytorch_cvt_convs as cvt
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

p = pathlib.Path(__file__).parent.resolve() / "../../../bmlogging"
sys.path.append(fspath(p))

import statistics
import time
from typing import Callable, List, Optional, Tuple

import loggerconstants

# FAMBench Logger
from bmlogger import get_bmlogger

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measuring CvT's Convolution Projection layer and Convolutional Token Embedding layer performance using PyTorch"
    )
    parser.add_argument("--warmups", type=int, default=10, help="warmup times")
    parser.add_argument("--steps", type=int, default=100, help="repeat times")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        required=True,
        help="valid devices",
    )
    parser.add_argument("--logger_file", type=str, default=None)
    parser.add_argument("--randomseed", type=int, default=0)

    parser.add_argument("--forward_only", dest="forward_only", action="store_true")
    parser.set_defaults(forward_only=False)
    args = parser.parse_args()

    # bmlogging header
    if args.logger_file is not None:
        bmlogger = get_bmlogger(log_file_path=args.logger_file)
    else:
        bmlogger = get_bmlogger(log_file_path=None)  # default to Nop logger

    # Tensor sizes used in this benchmark are taken from the CvT paper.
    # The authors trained CvT using input tensors of size 32 batches, 3 channels,
    # 224 rows, 224 columns.  Using that input size creates intermediate tensors
    # of sizes that are benchmarked here.

    # Note: conv_proj_k and conv_proj_v are identical in hyperparameters.
    # So conv_proj_k is benchmarked, while conv_proj_v is excluded as redundant.
    benchmark_cfgs_l = [
        {
            "layer_name": "cvt.stage0.block0.conv_proj_q",
            "input_shape": torch.Size([32, 64, 56, 56]),
            "kwargs": {
                "dim_in": 64,
                "dim_out": 64,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage0.block0.conv_proj_k\nSkip redundant benchmark of cvt.stage0.block0.conv_proj_v",
            "input_shape": torch.Size([32, 64, 56, 56]),
            "kwargs": {
                "dim_in": 64,
                "dim_out": 64,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block0.conv_proj_q",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block0.conv_proj_k\nSkip redundant benchmark of cvt.stage1.block0.conv_proj_v",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block1.conv_proj_q",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block1.conv_proj_k\nSkip redundant benchmark of cvt.stage1.block1.conv_proj_v)",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage2.blocks0.conv_proj_q\nSkip redundant benchmarks of cvt.stage2.blocks{1-9}.conv_proj_q",
            "input_shape": torch.Size([32, 384, 14, 14]),
            "kwargs": {
                "dim_in": 384,
                "dim_out": 384,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        # Blocks 0 through 9 in cvt.stage2 have identical hyperparameters
        {
            "layer_name": "cvt.stage2.blocks0.conv_proj_k\nSkip redundant benchmarks of cvt.stage2.blocks{1-9}.conv_proj_k and cvt.stage2.blocks{0-9}.conv_proj_v",
            "input_shape": torch.Size([32, 384, 14, 14]),
            "kwargs": {
                "dim_in": 384,
                "dim_out": 384,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage0.patch_embed",
            "input_shape": torch.Size([32, 3, 224, 224]),
            "kwargs": {
                "patch_size": 7,
                "in_chans": 3,
                "embed_dim": 64,
                "stride": 4,
                "padding": 2,
                "norm_layer": nn.LayerNorm,
            },
        },
        {
            "layer_name": "cvt.stage1.patch_embed",
            "input_shape": torch.Size([32, 64, 56, 56]),
            "kwargs": {
                "patch_size": 3,
                "in_chans": 64,
                "embed_dim": 192,
                "stride": 2,
                "padding": 1,
                "norm_layer": nn.LayerNorm,
            },
        },
        {
            "layer_name": "cvt.stage2.patch_embed",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "patch_size": 3,
                "in_chans": 192,
                "embed_dim": 384,
                "stride": 2,
                "padding": 1,
                "norm_layer": nn.LayerNorm,
            },
        },
    ]

    bench_time = []
    for cfg in benchmark_cfgs_l:
        layer_name = cfg["layer_name"]
        input_shape = cfg["input_shape"]
        kwargs = cfg["kwargs"]
        device = "cuda" if args.device == "gpu" else args.device
        if "conv_proj" in layer_name:
            layer_type = "conv_proj"
        elif "patch_embed" in layer_name:
            layer_type = "patch_embed"
        else:
            raise ValueError(f"{layer_name} is invalid.")

        if args.logger_file is not None:
            mode = "forward_only" if args.forward_only else "forward_and_backprop"
            bmlogger.header(
                "CVT", "UBENCH", "train", layer_name, score_metric=loggerconstants.GBPS
            )
            bmlogger.run_start()

        print("Benchmarking", layer_name)
        global_elap, global_bytes, global_flops = cvt.run(
            layer_type,
            input_shape,
            kwargs,
            device,
            args.warmups,
            args.steps,
            args.forward_only,
        )
        benchmark_metrics = {
            "GB/s": global_bytes / global_elap / 1.0e3,
            "TF/s": global_flops / global_elap / 1.0e12,
            "ELAP": global_elap,
            "FLOPS": global_flops,
        }
        print(benchmark_metrics)
        print("")
        if args.logger_file is not None:
            bmlogger.run_stop(
                args.steps, input_shape[0], extra_metadata=benchmark_metrics
            )
