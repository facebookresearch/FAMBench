#!/bin/bash

set -x
cd cudnn_multihead_attn

python multihead_attn_make_ref.py --batch-size=16 --double-precision-ref-data > /dev/null
./mha --iterations=100 --double_precision=true --training=true
python multihead_attn_make_ref.py --batch-size=64 --double-precision-ref-data > /dev/null
./mha --iterations=100 --double_precision=true --training=false

python multihead_attn_make_ref.py --batch-size=16 > /dev/null
./mha --iterations=100 --double_precision=false --training=true
python multihead_attn_make_ref.py --batch-size=64 > /dev/null
./mha --iterations=100 --double_precision=false --training=false
