#!/bin/bash

# Ideally this would be a script to run, log, and summarize all relevant benchmarks, 
# but at the moment it is more of an example. 

# DLRM OOTB
./run_dlrm_ootb_infer.sh -l results
./run_dlrm_ootb_train.sh -l results # ootb configs use config files. See docs/DLRM.md

# DLRM UBench
./run_dlrm_ubench_train_linear.sh -c "[(2,2,2,2,2)]" -l results # Config not real
./run_dlrm_ubench_train_embeddingbag.sh -l results -c "[(2,2,2,2),(2,2,2,2),(2,2,2,2),(2,2,2,2),(2,2,2,2)]" # Config not real

# XLMR OOTB
./run_xlmr_ootb.sh 

# view options: [raw_view -> pure json, intermediate_view -> nice table]
# intermediate view recommended for filling out table
python ../fb5logging/result_summarizer.py -f results -v intermediate_view