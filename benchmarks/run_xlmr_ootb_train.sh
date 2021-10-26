#!/bin/bash

# Default values
benchmark=xlmr
implementation=ootb
mode=train
config=fb-1dev-short
LOG_DIR=results
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"
nbatches=10

python3 "${benchmark}/${implementation}/xlmr.py" --logfile=${LOGGER_FILE} --num-batches=${nbatches} --famconfig=${config} --use-gpu
