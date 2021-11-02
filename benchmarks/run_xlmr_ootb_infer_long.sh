#!/bin/bash

# Default values
benchmark=xlmr
implementation=ootb
mode=eval
config=fb-1dev-long
LOG_DIR=results
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"
nbatches=100
batchsize=32
seqlength=256
vocabsize=250000

python "${benchmark}/${implementation}/xlmr.py" --inference-only ${config_flags} --logfile=${LOGGER_FILE} --num-batches=${nbatches} --batch-size=${batchsize} --sequence-length=${seqlength} --vocab-size=${vocabsize} --famconfig=${config} --half-model --use-gpu
