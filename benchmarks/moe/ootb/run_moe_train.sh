#!/usr/bin/env bash

set -ex

GPUS=${1:-8}
MAX_TOKENS=${2:-16384}

export NODE_COUNT=${NODE_COUNT:=1}
export NODE_RANK=${RANK:=0}
export MASTER_ADDR=${MASTER_ADDR:=127.0.0.1}
export MASTER_PORT=${MASTER_PORT:=9000}

export MAX_TOKENS=${MAX_TOKENS}
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $[${GPUS}-1])
export DATABIN=$(realpath MOE_DATA/wmt16_en_de/databin)

OUTPUT_DIR=MOE_OUTPUT
mkdir -p ${OUTPUT_DIR}
OUTPUT_DIR=$(realpath ${OUTPUT_DIR})
LOG_FILE=${OUTPUT_DIR}/moe_${NODE_COUNT}x${GPUS}x${MAX_TOKENS}.log

export WORKSPACE=${OUTPUT_DIR}

export NUM_GPUS=${GPUS}

# Set EP_WORLD_SIZE and NUM_EXPERTS to min(8, NUM_GPUS)
if [ $NUM_GPUS -gt 8 ]
then
    export EP_WORLD_SIZE=8
    export NUM_EXPERTS=8
else
    export EP_WORLD_SIZE=${NUM_GPUS}
    export NUM_EXPERTS=${NUM_GPUS}
fi
export ARCH=transformer_ds_moe_vaswani_wmt_en_de_big
export HSA_ENABLE_SDMA=0
unset WORLD_SIZE

cd fairseq/examples/deepspeed/moe_e/

# Shorten the training loop (make sure you finish a complete epoch)
max_update=$[80 / ${NODE_COUNT}]
sed -i "s/max-update 300000/max-update ${max_update}/g" run-distributed.sh
sed -i "s/--validate-interval-updates 20/--log-interval 1 --disable-validation/g" run-distributed.sh
sed -i "s/--save-interval-updates 1/--save-interval-updates 0 --save-interval 1000/g" run-distributed.sh

bash run-distributed.sh 2>&1 | tee ${LOG_FILE}

cd ${OUTPUT_DIR}
python ../moe_parser.py --logpath ${LOG_FILE} --select_epoch 0
