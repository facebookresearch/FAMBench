#!/bin/bash

set -x
pushd cudnn_multihead_attn

nvcc -o mha -I/usr/include -L/usr/lib/x86_64-linux-gnu -l:libcudnn.so.8.6.0 -l:libsqlite3.so.0 -l:libgflags.so -lpthread -ldl --std=c++17 cudnn_multihead_attn_benchmark.cu

python multihead_attn_make_ref.py --batch-size=16 --double-precision-ref-data > /dev/null
./mha --iterations=100 --double_precision=true --training=true
python multihead_attn_make_ref.py --batch-size=64 --double-precision-ref-data > /dev/null
./mha --iterations=100 --double_precision=true --training=false

python multihead_attn_make_ref.py --batch-size=16 > /dev/null
./mha --iterations=100 --double_precision=false --training=true
python multihead_attn_make_ref.py --batch-size=64 > /dev/null
./mha --iterations=100 --double_precision=false --training=false

popd
