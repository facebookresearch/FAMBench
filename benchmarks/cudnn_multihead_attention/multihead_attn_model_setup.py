# Verify extensional equality.

import torch
from torch import nn
import math
import sqlite3
import os
import sys
import argparse
import pendulum
import time
import json
import struct
import copy

def save_to_sqlite_db(args, net_initial, x, y_pred, y_true):

    db_filename = "multihead_attn_model_data.db"
    if os.path.isfile(db_filename):
        os.remove(db_filename)
    conn = sqlite3.connect(db_filename)
    conn.execute('pragma journal_mode=wal')
    sql_c = conn.cursor()
    sql_c.execute('''CREATE TABLE IF NOT EXISTS TestData (dataset_id TEXT, name TEXT, data blob)''')
    sql_c.execute('''CREATE INDEX IF NOT EXISTS dataset_id_index ON TestData (dataset_id)''')

    data_sets_l = list(net_initial.named_parameters())
    data_sets_l += [("x", x), ("y_pred", y_pred), ("y_true", y_true)]

    args.params_count = sum(p.numel() for p in net_initial.parameters())

    dataset_id = 'Dataset_id ' + str(pendulum.from_timestamp(time.time()).in_timezone('GMT'))[0:19]   

    sql_c.execute("BEGIN") 
    for key, val in vars(args).items():
        db_record = (dataset_id, key, float(val))    
        sql_c.execute('INSERT INTO TestData VALUES (?,?,?)', db_record)
    for group in data_sets_l:
        name = group[0]
        data = group[1].data.cpu().numpy()
        ba = bytearray()
        for val in data.flatten():
            ba += struct.pack("f", float(val))
        db_record = (dataset_id, name, ba)
        sql_c.execute('INSERT INTO TestData VALUES (?,?,?)', db_record)    
    conn.commit() 
    conn.close()

class MultiHeadAttention(nn.Module):
    def __init__(self, nheads, emb_dim, dropout = 0.0, use_bias = True, batch_first=False):
        super().__init__()

        self.nheads = nheads 
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.sub_emb_dim = emb_dim // nheads        

        self.q_p = nn.Linear(emb_dim, emb_dim, bias=use_bias)
        self.k_p = nn.Linear(emb_dim, emb_dim, bias=use_bias) 
        self.v_p = nn.Linear(emb_dim, emb_dim, bias=use_bias)       
        self.o_p = nn.Linear(emb_dim, emb_dim, bias=use_bias)        
        
        # Set weights manually for debugging purposes.
        # self.q_p.weight.data[:] = 1
        self.q_p.weight.data[:] = torch.arange(emb_dim * emb_dim).reshape(emb_dim, emb_dim)[:]
        # self.q_p.weight.data=torch.eye(emb_dim)
        # self.k_p.weight.data[:] = 1
        self.k_p.weight.data[:] = torch.arange(emb_dim * emb_dim).reshape(emb_dim, emb_dim)[:]
        # self.k_p.weight.data=torch.eye(emb_dim)
        # self.v_p.weight.data[:] = 1
        self.v_p.weight.data[:] = torch.arange(emb_dim * emb_dim).reshape(emb_dim, emb_dim)[:]
        # self.v_p.weight.data=torch.eye(emb_dim)        
        # self.o_p.weight.data=torch.eye(emb_dim)
        self.o_p.weight.data[:] = 1.0
        #self.o_p.weight.data[:] = torch.arange(emb_dim * emb_dim).reshape(emb_dim, emb_dim)[:]
    
    def forward(self, q, k, v, need_weights=False, mask=None):        
        if self.batch_first:
            batch_size, seq_len, emb_dim = q.shape         
        else:
            seq_len, batch_size, emb_dim = q.shape         
            k = k.transpose(0,1)
            q = q.transpose(0,1)
            v = v.transpose(0,1)
        q = self.q_p(q).view(batch_size, seq_len, self.nheads, self.sub_emb_dim).permute(0,2,1,3)
        k = self.k_p(k).view(batch_size, seq_len, self.nheads, self.sub_emb_dim).permute(0,2,3,1)
        v = self.v_p(v).view(batch_size, seq_len, self.nheads, self.sub_emb_dim).permute(0,2,1,3)
        
        scores = torch.matmul(q, k) / math.sqrt(self.sub_emb_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = nn.functional.softmax(scores, dim=-1)
        #if self.dropout is not None:
        #    scores = self.dropout(scores)

        scores = torch.matmul(scores, v)

        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_len, emb_dim)        
        output = self.o_p(concat)

        if not self.batch_first:
            output = output.permute(1,0,2)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-head Attention model configuration setup."
    )
    parser.add_argument("--emb-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--disable-bias", action="store_true", default=True)
    # Dropout is not supposed by CUDNN's multihead attn implementation.
    parser.add_argument("--dropout", type=float, default=0.0)
    
    args = parser.parse_args()
    args.use_bias = not args.disable_bias

    # _sf means tensor shape format is sequence first.
    # _bf means tensor shape format is batch first.
    pyt_nn_sf = nn.MultiheadAttention(
        embed_dim=args.emb_dim,
        num_heads=args.num_heads, 
        dropout=args.dropout, 
        bias=args.use_bias, 
        add_bias_kv=args.use_bias, 
        add_zero_attn=False, 
        kdim=args.emb_dim, 
        vdim=args.emb_dim, 
        batch_first=False,
        device=None,
    )
    pyt_nn_bf = nn.MultiheadAttention(
        embed_dim=args.emb_dim,
        num_heads=args.num_heads, 
        dropout=args.dropout, 
        bias=args.use_bias, 
        add_bias_kv=args.use_bias, 
        add_zero_attn=False, 
        kdim=args.emb_dim, 
        vdim=args.emb_dim, 
        batch_first=True,
        device=None,
    )    
    selfcoded_nn_sf = MultiHeadAttention(
        nheads=args.num_heads, 
        emb_dim=args.emb_dim, 
        dropout=args.dropout,
        use_bias=args.use_bias,
        batch_first=False,
    )
    selfcoded_nn_bf = MultiHeadAttention(
        nheads=args.num_heads, 
        emb_dim=args.emb_dim, 
        dropout=args.dropout,
        use_bias=args.use_bias,
        batch_first=True,
    )    
    selfcoded_nn_bf.k_p.weight.data = selfcoded_nn_sf.k_p.weight.data.clone()
    selfcoded_nn_bf.q_p.weight.data = selfcoded_nn_sf.q_p.weight.data.clone()
    selfcoded_nn_bf.v_p.weight.data = selfcoded_nn_sf.v_p.weight.data.clone()
    selfcoded_nn_bf.o_p.weight.data = selfcoded_nn_sf.o_p.weight.data.clone()

    pyt_nn_total_params = sum(p.numel() for p in pyt_nn_sf.parameters())
    print(f"Built-in Pytorch Imp, Total Parameters: {pyt_nn_total_params}")

    selfcoded_nn_total_params = sum(p.numel() for p in selfcoded_nn_sf.parameters())
    print(f"Manual Imp, Total Parameters: {selfcoded_nn_total_params}")

    # Copy parameter values so both nets have the same parameter values.
    pyt_nn_sf.in_proj_weight.data = torch.cat([list(selfcoded_nn_sf.named_parameters())[i][1].clone() for i in range(3)])
    pyt_nn_sf.out_proj.weight.data = selfcoded_nn_sf.o_p.weight.clone()

    pyt_nn_bf.in_proj_weight.data = torch.cat([list(selfcoded_nn_sf.named_parameters())[i][1].clone() for i in range(3)])
    pyt_nn_bf.out_proj.weight.data = selfcoded_nn_sf.o_p.weight.clone()    

    x_sf = torch.randn((args.seq_len, args.batch_size, args.emb_dim), requires_grad=False)
    # Optionally set values for debugging purposes    
    if False:
        for sl in range(args.seq_len):
            for bs in range(args.batch_size):
                for em in range(args.emb_dim):
                    x_sf.data[sl,bs,em] = sl + bs*5 + em*20

    x_bf = torch.randn((args.batch_size, args.seq_len, args.emb_dim), requires_grad=False)
    # Optionally set values for debugging purposes
    if False:
        for sl in range(args.seq_len):
            for bs in range(args.batch_size):
                for em in range(args.emb_dim):
                    x_bf.data[bs,sl,em] = sl + bs*5 + em*20

    #use .permute(1,0,2) to convert one to match the other.
    y_pred_sf_pyt, _1 = pyt_nn_sf(x_sf,x_sf,x_sf, need_weights=False)
    y_pred_bf_pyt, _2 = pyt_nn_bf(x_bf,x_bf,x_bf, need_weights=False)

    y_pred_sf_man = selfcoded_nn_sf(x_sf,x_sf,x_sf, need_weights=False)
    y_pred_bf_man = selfcoded_nn_bf(x_bf,x_bf,x_bf, need_weights=False)
    
    if torch.any( abs(y_pred_sf_pyt - y_pred_sf_man)/y_pred_sf_man > 0.1):
        print("x_sf: ",x_sf)
        print("y_pred_sf_man: ",y_pred_sf_man)
        print("y_pred_sf_pyt: ",y_pred_sf_pyt)        
        print("Sequence-first version, Failure! MISMATCH.")
    else:         
        print("Using sequence-first format, Success! Manual implementation matches PyTorch's built-in implementation.")   

    if torch.any( abs(y_pred_bf_pyt - y_pred_bf_man)/y_pred_bf_man > 0.1):
        print("x_bf: ",x_bf)
        print("y_pred_bf_man: ",y_pred_bf_man)
        print("y_pred_bf_pyt: ",y_pred_bf_pyt)        
        print("Batch-first version, Failure! MISMATCH.")
    else:          
        print("Using batch-first format, Success! Manual implementation matches PyTorch's built-in implementation.")          

    y_true_sf = torch.randn_like(y_pred_sf_man)
    y_true_bf = torch.randn_like(y_pred_bf_man)
    net_initial = selfcoded_nn_sf

    # Transpose because CUDNN uses the transposed format.
    net_initial.q_p.weight.data = net_initial.q_p.weight.data.transpose(0,1)
    net_initial.k_p.weight.data = net_initial.k_p.weight.data.transpose(0,1)
    net_initial.v_p.weight.data = net_initial.v_p.weight.data.transpose(0,1)
    net_initial.o_p.weight.data = net_initial.o_p.weight.data.transpose(0,1)

    # save db info
    save_to_sqlite_db(args, net_initial, x_sf, y_pred_sf_man, y_true_sf)    
    print("Done.")
