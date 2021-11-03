#!/bin/bash

# Default values
benchmark=xlmr
implementation=ootb
LOG_DIR=results
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"


mode=train
config=msft-1dev

nbatches=10
batchsize=4
seqlength=512
vocabsize=250000

python "${benchmark}/${implementation}/xlmr.py" --logfile=${LOGGER_FILE} --num-batches=${nbatches} --batch-size=${batchsize} --sequence-length=${seqlength} --vocab-size=${vocabsize} --famconfig=${config} --half-model --use-gpu
