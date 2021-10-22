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
benchmark=rnnt
implementation=ootb
mode=eval
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

# Resolve to absolute directory to support both relative and absolute dirs.
ABSOLUTE_LOG_DIR=`readlink -f ${LOG_DIR}`
LOGGER_FILE="${ABSOLUTE_LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
echo "Config: ${config}"
echo "Saving FB5 Logger File: ${LOGGER_FILE}"
echo
echo "Running Command:"

if [[ -z "$DATASET_DIR" ]]; then
  echo "ERROR: DATASET_DIR not set!"
  exit 1
fi
if [[ -z "$RESULT_DIR" ]]; then
  echo "ERROR: RESULT_DIR not set!"
  exit 1
fi

benchmark_root=${benchmark}/${implementation}/inference
if [ "$is_config" = true ]; then
  config_flags=$(head -n 1 "${config}")
  (set -x; python ${benchmark_root}/run.py --backend pytorch --dataset_dir ${DATASET_DIR}/LibriSpeech --manifest ${DATASET_DIR}/LibriSpeech/librispeech-dev-clean-wav.json --pytorch_config_toml ${benchmark_root}/pytorch/configs/rnnt.toml --pytorch_checkpoint ${RESULT_DIR}/rnnt.pt --scenario Offline --log_dir ${RESULT_DIR}/Offline_pytorch_rerun --fb5logger ${LOGGER_FILE} --fb5config ${config} ${config_flags} 2>&1)
else
  (set -x; python ${benchmark_root}/run.py --backend pytorch --dataset_dir ${DATASET_DIR}/LibriSpeech --manifest ${DATASET_DIR}/LibriSpeech/librispeech-dev-clean-wav.json --pytorch_config_toml ${benchmark_root}/pytorch/configs/rnnt.toml --pytorch_checkpoint ${RESULT_DIR}/rnnt.pt --scenario Offline --log_dir ${RESULT_DIR}/Offline_pytorch_rerun --fb5logger ${LOGGER_FILE} --fb5config ${config} 2>&1)
fi

echo "=== Completed Run ==="
