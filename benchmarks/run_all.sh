#!/bin/bash

# DLRM
./run_dlrm_ootb_infer.sh -l results
./run_dlrm_ootb_train.sh -l results
./run_dlrm_ubench_train_embeddingbag.sh -l results
./run_dlrm_ubench_train_linear.sh -l results

# view options: [raw_view, intermediate_view]
python ../fb5logging/result_summarizer.py -f results -v intermediate_view