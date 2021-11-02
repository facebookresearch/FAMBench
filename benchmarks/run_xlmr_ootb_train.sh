#!/bin/bash

# Default values
benchmark=xlmr
implementation=ootb
mode=train
config=msft-1dev
LOG_DIR=results
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"
nbatches=25
batchsize=32
seqlength=64
vocabsize=250000

python "${benchmark}/${implementation}/xlmr.py" --logfile=${LOGGER_FILE} --num-batches=${nbatches} --batch-size=${batchsize} --sequence-length=${seqlength} --vocab-size=${vocabsize} --famconfig=${config} --use-gpu
