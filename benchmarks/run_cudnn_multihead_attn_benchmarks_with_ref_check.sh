#!/bin/bash

set -x
cd cudnn_multihead_attn

python multihead_attn_make_ref.py --batch-size=16 --double-precision-ref-data
./mha --iterations=100 --double_precision=true --training=true --print_accuracy_stats=true --debug_mode=true
python multihead_attn_make_ref.py --batch-size=64 --double-precision-ref-data
./mha --iterations=100 --double_precision=true --training=false --print_accuracy_stats=true --debug_mode=true

python multihead_attn_make_ref.py --batch-size=16
./mha --iterations=100 --double_precision=false --training=true --print_accuracy_stats=true --debug_mode=true
python multihead_attn_make_ref.py --batch-size=64
./mha --iterations=100 --double_precision=false --training=false --print_accuracy_stats=true --debug_mode=true
