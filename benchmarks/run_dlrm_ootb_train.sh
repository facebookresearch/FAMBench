#!/bin/bash

printUsage() {
  echo
  echo "Usage: $(basename "$0") <options>"
  echo
  echo "Options:"
  echo "  -h                      Prints this help."
  echo "  -l <dir to save log>    Saves FB5 Log to specified directory in first argument."
  echo "  -c <config file>        Runs the command in the config file instead of the default config."
  echo
  return 0
}

if [ "$1" == "" ]; then
  printUsage
  exit 0
fi

# Default values
benchmark=dlrm
implementation=ootb
mode=train
config=tiny # default is tiny, proof of concept
is_config=false

while getopts "hl:c:" flag ;
do
  case "${flag}" in
    h)
      printUsage ; exit 0 ;;
    l)
      LOG_DIR=${OPTARG} ;;
    c)
      config=${OPTARG} ; is_config=true ;;
  esac
done

LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
echo "Config: ${config}"
echo "Saving FB5 Logger File: ${LOGGER_FILE}"
echo
echo "Running Command:"

if [ "$is_config" = true ]; then
  config_flags=$(head -n 1 "${config}")
  (set -x; python "${benchmark}/${implementation}/dlrm_s_pytorch.py" ${config_flags} --fb5logger=${LOGGER_FILE} --fb5config=${config} 2>&1)
else
  (set -x; python "${benchmark}/${implementation}/dlrm_s_pytorch.py" --mini-batch-size=64 --test-mini-batch-size=64 --test-num-workers=0 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --numpy-rand-seed=727 --num-batches=200 --print-freq=20 --print-time --fb5logger=${LOGGER_FILE} --fb5config=${config} 2>&1)
fi

echo "=== Completed Run ==="
