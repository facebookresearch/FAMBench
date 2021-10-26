#!/bin/bash

# Default values
benchmark=xlmr
implementation=ootb
mode=eval
config=fb-1dev-short
LOG_DIR=results
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"
nbatches=100

python "${benchmark}/${implementation}/xlmr.py" --inference-only ${config_flags} --logfile=${LOGGER_FILE} --num-batches=${nbatches} --famconfig=${config} --use-gpu
