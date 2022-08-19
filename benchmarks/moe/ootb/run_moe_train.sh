set -ex
NODES=${1:-1}
GPUS=${2:-8}
MAX_TOKENS=${3:-16384}

export MAX_TOKENS=${MAX_TOKENS}
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $[${GPUS}-1])
export DATABIN=$(realpath MOE_DATA/wmt16_en_de/databin)

OUTPUT_DIR=MOE_OUTPUT
mkdir -p ${OUTPUT_DIR}
OUTPUT_DIR=$(realpath ${OUTPUT_DIR})
LOG_FILE=${OUTPUT_DIR}/moe_${NODES}x${GPUS}x${MAX_TOKENS}.log

export WORKSPACE=${OUTPUT_DIR}

export NUM_GPUS=${GPUS}
export EP_WORLD_SIZE=8
export NUM_EXPERTS=8
export NODE_COUNT=${NODES}
export ARCH=transformer_ds_moe_vaswani_wmt_en_de_big
export HSA_ENABLE_SDMA=0
unset WORLD_SIZE

cd fairseq/examples/deepspeed/moe_e/

# Shorten the training loop
max_update=$[80 / ${NODES}]
sed -i "s/max-update 300000/max-update ${max_update}/g" run-distributed.sh
sed -i "s/--validate-interval-updates 20/--log-interval 1 --disable-validation/g" run-distributed.sh
sed -i "s/--save-interval-updates 1/--save-interval-updates 0 --save-interval 1000/g" run-distributed.sh

bash run-distributed.sh 2>&1 | tee ${LOG_FILE}

cd ${OUTPUT_DIR}
python ../moe_parser.py --logpath ${LOG_FILE} --select_epoch 0
