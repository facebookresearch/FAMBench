#!/usr/bin/env bash

set -ex

GPUS=${1:-8}
BS=${2:-256}

export NODE_COUNT=${NODE_COUNT:=1}
export RANK=${RANK:=0}
export MASTER_ADDR=${MASTER_ADDR:=127.0.0.1}
export MASTER_PORT=${MASTER_PORT:=9000}

cd CvT

OUTPUT_DIR=../OUTPUT
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/cvt_${NODE_COUNT}x${GPUS}x${BS}.log

bash run-alt.sh -g ${GPUS} -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml \
    DATASET.ROOT ../DATASET/imagenet OUTPUT_DIR ${OUTPUT_DIR} \
    TRAIN.END_EPOCH 300 TRAIN.BATCH_SIZE_PER_GPU ${BS} 2>&1 | tee ${LOG_FILE}

cd ${OUTPUT_DIR}
python ../cvt_parser.py --logpath ${LOG_FILE} --steps $[${GPUS}*10]
