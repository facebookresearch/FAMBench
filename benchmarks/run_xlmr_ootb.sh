#!/bin/bash

# fixed values
benchmark=xlmr
implementation=ootb

# default values
config_name=default-config
nbatches=10
batchsize=16
seqlength=16
vocabsize=250000
LOG_DIR=results
config_flags="--inference-only --num-batches=${nbatches} --batch-size=${batchsize} --sequence-length=${seqlength} --vocab-size=${vocabsize} --famconfig=${config_name} --half-model --use-gpu"

while getopts "hl:c:" flag ;
do
  case "${flag}" in
    h)
      printUsage ; exit 0 ;;
    l)
      LOG_DIR=${OPTARG} ;;
    c)
      config_flags=${OPTARG} ;;
  esac
done

(set -x; python "${benchmark}/${implementation}/xlmr.py" ${config_flags} --logdir=${LOG_DIR})
