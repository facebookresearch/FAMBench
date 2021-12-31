#!/bin/bash

./run_xlmr_ootb.sh -c "--inference-only --famconfig=fb-1dev-A --num-batches=100 --batch-size=96 --sequence-length=64 --vocab-size=250000 --half-model --use-gpu --warmup-batches=20"
./run_xlmr_ootb.sh -c "--inference-only --famconfig=fb-1dev-A-realdist --num-batches=100 --batch-size=64 --vocab-size=250000 --half-model --use-gpu --warmup-batches=20 --seqlen-dist=[[1,2462],[0.99,675],[0.95,250],[0.9,147],[0.75,56],[0.7,47],[0.5,23],[0.25,11],[0.05,3],[0,1]] --seqlen-dist-max=256"
./run_xlmr_ootb.sh -c "--inference-only --famconfig=fb-1dev-B --num-batches=100 --batch-size=64 --sequence-length=256 --vocab-size=250000 --half-model --use-gpu --warmup-batches=20"
./run_xlmr_ootb.sh -c "--famconfig=fb-1dev-A-fp32 --num-batches=50 --batch-size=32 --sequence-length=256 --vocab-size=250000 --use-gpu --warmup-batches=10"
./run_xlmr_ootb.sh -c "--famconfig=fb-1dev-A-fp16 --num-batches=50 --batch-size=32 --sequence-length=256 --vocab-size=250000 --half-model --use-gpu --warmup-batches=10"
./run_xlmr_ootb.sh -c "--famconfig=msft-1dev --num-batches=50 --batch-size=16 --sequence-length=512 --vocab-size=250000 --half-model --use-gpu --warmup-batches=10"
# view options: [raw_view -> pure json, intermediate_view -> nice table]
# intermediate view recommended for filling out table
python ../bmlogging/result_summarizer.py -f results -v intermediate_view