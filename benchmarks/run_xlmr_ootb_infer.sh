#!/bin/bash

# Default values
benchmark=xlmr
implementation=ootb
mode=eval
config=tiny # default is tiny, proof of concept
LOG_DIR=results
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

python "${benchmark}/${implementation}/xlmr.py" --inference-only ${config_flags} --logfile=${LOGGER_FILE} --fb5config=${config}