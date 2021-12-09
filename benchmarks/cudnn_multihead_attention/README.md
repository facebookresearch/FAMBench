# Multihead Attention CUDNN benchmark

## Requirements to compile and run cudnn_multihead_attn_benchmark.cu
CUDNN 8
CUDA Toolkit 11
Sqlite 3
gflags

## Example Usage:
./cudnn_multihead_attn_benchmark --training=true --batchsize=16 --iterations=100 --modelconfig=./multihead_attn_model_data.db

## Reconfiguring the benchmark
run: python multihead_attn_model_setup.py --emb-dim=<number> --num-heads=<number> --seq-len=<number>
This will delete and create anew multihead_attn_model_data.db containing the new config information and model data.
