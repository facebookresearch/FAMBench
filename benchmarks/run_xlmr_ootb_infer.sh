#!/bin/bash

# Default values
benchmark=xlmr
implementation=ootb
mode=eval
config=fb-1dev-short
LOG_DIR=results
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

<<<<<<< HEAD
python "${benchmark}/${implementation}/xlmr.py" --inference-only ${config_flags} --logfile=${LOGGER_FILE} --fb5config=${config}
=======
python "${benchmark}/${implementation}/xlmr.py" --inference-only ${config_flags} --logfile=${LOGGER_FILE} --famconfig=${config} --use-gpu
>>>>>>> 74dc5d29c238c013643fdcc62d7a851a02467eb3
