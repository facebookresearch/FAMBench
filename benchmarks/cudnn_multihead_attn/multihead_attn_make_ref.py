# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This script generates reference data that is used to benchmark
# CuDNN's Multi-head Attention training and inference implementations.
# The reference data is saved to the file "multihead_attn_ref_data.db", which is
# read by cudnn_multihead_attn_benchmark.cu.

import argparse
import datetime
import math
import os
import random
import sqlite3
import struct
import pathlib

import torch
from torch import nn


def save_to_sqlite_db(args, net_initial, q, k, v, o):

    # Add weights & biases tensors to dataset.
    dataset_l = list(net_initial.named_parameters())
    # Add gradients tensors to dataset.
    dataset_l += [(k + ".grad", t.grad) for k, t in net_initial.named_parameters()]
    # Add sample input-output tensors to dataset.
    dataset_l += [("q", q), ("k", k), ("v", v), ("o", o)]

    # Add total parameters count to args so it's stored in the db.
    args.params_count = sum(p.numel() for p in net_initial.parameters())

    db_filename = pathlib.Path(__file__).parent.resolve() / "multihead_attn_ref.db"
    if db_filename.is_file():
        db_filename.unlink()

    print(f"Saving reference to file: {db_filename}.")

    dataset_id = "Dataset_id " + datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )    
    conn = sqlite3.connect(db_filename)
    conn.execute("pragma journal_mode=wal")
    sql_c = conn.cursor()
    sql_c.execute(
        """CREATE TABLE IF NOT EXISTS RefData (dataset_id TEXT, name TEXT, data blob)"""
    )
    sql_c.execute(
        """CREATE INDEX IF NOT EXISTS dataset_id_index ON RefData (dataset_id)"""
    )

    sql_c.execute("BEGIN")

    # (Insert single values into db.)
    # Insert argparse variables.
    for key, val in vars(args).items():
        db_record = (dataset_id, key, val)
        sql_c.execute("INSERT INTO RefData VALUES (?,?,?)", db_record)

    # (Insert tensors into db.)
    # Insert neural net weights, biases, sample q, k, v, o activations,
    # and respective gradients.
    struct_format = "d" if args.double_precision_ref_data else "f"
    for key, tensor in dataset_l:
        ba = bytearray()
        for val in tensor.cpu().detach().numpy().flatten():
            ba += struct.pack(struct_format, val)
        db_record = (dataset_id, key, ba)
        sql_c.execute("INSERT INTO RefData VALUES (?,?,?)", db_record)

    conn.commit()
    conn.close()


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        nheads,
        emb_dim,
        dropout=0.0,
        use_bias=True,
        batch_first=False,
        dtype=torch.float64,
    ):
        super().__init__()

        self.nheads = nheads
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.sub_emb_dim = emb_dim // nheads

        self.q_p = nn.Linear(emb_dim, emb_dim, bias=use_bias, dtype=dtype)
        self.k_p = nn.Linear(emb_dim, emb_dim, bias=use_bias, dtype=dtype)
        self.v_p = nn.Linear(emb_dim, emb_dim, bias=use_bias, dtype=dtype)
        self.o_p = nn.Linear(emb_dim, emb_dim, bias=use_bias, dtype=dtype)

    def forward(self, q, k, v, need_weights=False, mask=None):
        if self.batch_first:
            batch_size, seq_len, emb_dim = q.shape
        else:
            seq_len, batch_size, emb_dim = q.shape
            k = k.transpose(0, 1)
            q = q.transpose(0, 1)
            v = v.transpose(0, 1)

        q = self.q_p(q)
        q = q.view(batch_size, seq_len, self.nheads, self.sub_emb_dim)
        q = q.permute(0, 2, 1, 3)
        
        k = self.k_p(k)
        k = k.view(batch_size, seq_len, self.nheads, self.sub_emb_dim)
        k = k.permute(0, 2, 3, 1)

        scores = torch.matmul(q, k) / math.sqrt(self.sub_emb_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = nn.functional.softmax(scores, dim=-1)
        if self.dropout is not None:
            scores = self.dropout(scores)

        v = self.v_p(v)
        v = v.view(batch_size, seq_len, self.nheads, self.sub_emb_dim)
        v = v.permute(0, 2, 1, 3)   
             
        scores = torch.matmul(scores, v)
        
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        output = self.o_p(concat)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output


if __name__ == "__main__":
    print("\nGenerating multi-head attention reference data. " +
        "(Model spec. and tensors; sample q, k, v, o activations; and gradients.)")
    parser = argparse.ArgumentParser(
        description="Multi-head Attention reference model and data setup."
    )
    parser.add_argument("--emb-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    
    parser.add_argument("--double-precision-ref-data", action="store_true", default=False)

    parser.add_argument("--qkv-max-sig-figs", type=int, default=7)
    parser.add_argument("--qkv-highest-val", type=float, default=10000)
    parser.add_argument("--qkv-lowest-val", type=float, default=0)

    # Gradients for biases are not computed in CuDNN 8.3.0's multihead attn. imp. (MHA),
    # even though the API reference indicates they should be computed. (CuDNN Bug?)
    parser.add_argument("--disable-bias", action="store_true", default=True)
    # Dropout for MHA is not supported by CuDNN 8.3.0, as stated in its API reference.
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    args.use_bias = not args.disable_bias
    dtype = torch.float64 if args.double_precision_ref_data else torch.float32
    pyt_net = nn.MultiheadAttention(
        embed_dim=args.emb_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        bias=args.use_bias,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=args.emb_dim,
        vdim=args.emb_dim,
        batch_first=False,
        device=None,
        dtype=dtype,
    )
    selfcoded_net = MultiHeadAttention(
        nheads=args.num_heads,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        use_bias=args.use_bias,
        batch_first=False,
        dtype=dtype,
    )
    pyt_net_total_params = sum(p.numel() for p in pyt_net.parameters())
    selfcoded_net_total_params = sum(p.numel() for p in selfcoded_net.parameters())
    if pyt_net_total_params != selfcoded_net_total_params:
        print("Error. Number of total parameters mismatch.")
        print(f"PyTorch's Multi-head Attention, Total Parameters: {pyt_net_total_params}")
        print(f"Self-coded Multi-head Attention, Total Parameters: {selfcoded_net_total_params}")        
        exit()
    print(f"Number of Trainable Parameters: {pyt_net_total_params}")
    
    # Copy all weights & biases so both nets have the same parameter values.
    # Copy weights & biases for queries, keys, and values linear layers.
    pyt_net.in_proj_weight.data = torch.cat(
        [
            selfcoded_net.q_p.weight.data,
            selfcoded_net.k_p.weight.data,
            selfcoded_net.v_p.weight.data,
        ]
    ).clone()
    if args.use_bias:
        pyt_net.in_proj_bias.data = torch.cat(
            [
                selfcoded_net.q_p.bias.data,
                selfcoded_net.k_p.bias.data,
                selfcoded_net.v_p.bias.data,
            ]
        ).clone()
    # Copy weights & biases for the output linear layer.
    pyt_net.out_proj.weight.data = selfcoded_net.o_p.weight.clone()
    if args.use_bias:
        pyt_net.out_proj.bias.data = selfcoded_net.o_p.bias.clone()

    # Make sample q, k, v input tensors to benchmark with.
    low, high = args.qkv_lowest_val, args.qkv_highest_val
    max_sig_figs = args.qkv_max_sig_figs

    x = torch.rand((args.seq_len, args.batch_size, args.emb_dim), dtype=dtype)
    x = x * (high-low) + low 
    least_sig_place = math.ceil(math.log(max(abs(high),abs(low)))/math.log(10)) - max_sig_figs 
    x = torch.trunc(x / 10**least_sig_place) * (10**least_sig_place)

    print("Example q, k, v values:", x.flatten().numpy()[:10])
    q = x.clone()
    k = x.clone()
    v = x.clone()
    o_pyt, _ = pyt_net(q, k, v, need_weights=False)
    o_selfcoded = selfcoded_net(q, k, v, need_weights=False)

    if torch.any(abs(o_pyt - o_selfcoded) / o_selfcoded > 0.01):
        print("o_selfcoded: ", o_selfcoded)
        print("o_pyt: ", o_pyt)
        print(
            "Error. Results mismatch between self-coded multi-head attention and PyTorch's imp."
        )
        exit()

    gradient_signal = torch.ones_like(o_selfcoded)
    o_selfcoded.backward(gradient_signal)

    # Transpose to CuDNN 8.3.0's format.
    selfcoded_net.q_p.weight.data = selfcoded_net.q_p.weight.data.transpose(0, 1)
    selfcoded_net.k_p.weight.data = selfcoded_net.k_p.weight.data.transpose(0, 1)
    selfcoded_net.v_p.weight.data = selfcoded_net.v_p.weight.data.transpose(0, 1)
    selfcoded_net.o_p.weight.data = selfcoded_net.o_p.weight.data.transpose(0, 1)
    selfcoded_net.q_p.weight.grad.data = selfcoded_net.q_p.weight.grad.data.transpose(0, 1)
    selfcoded_net.k_p.weight.grad.data = selfcoded_net.k_p.weight.grad.data.transpose(0, 1)
    selfcoded_net.v_p.weight.grad.data = selfcoded_net.v_p.weight.grad.data.transpose(0, 1)
    selfcoded_net.o_p.weight.grad.data = selfcoded_net.o_p.weight.grad.data.transpose(0, 1)    

    # save db info
    save_to_sqlite_db(args, selfcoded_net, q, k, v, o_selfcoded)
    print("\nDone.\n")
