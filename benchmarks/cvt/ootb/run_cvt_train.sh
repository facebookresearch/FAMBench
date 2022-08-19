NODES=${1:-1}
GPUS=${2:-8}
BS=${3:-256}

NODE_RANK=${4:-0}
MASTER_ADDR=${5:-127.0.0.1}

export NODE_COUNT=${NODES}
export RANK=${NODE_RANK}
export MASTER_ADDR=${MASTER_ADDR}

set -ex

cd CvT

OUTPUT_DIR=../OUTPUT
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/cvt_${NODES}x${GPUS}x${BS}.log

bash run-alt.sh -g ${GPUS} -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml \
    DATASET.ROOT ../DATASET/imagenet OUTPUT_DIR ${OUTPUT_DIR} \
    TRAIN.END_EPOCH 300 TRAIN.BATCH_SIZE_PER_GPU ${BS} 2>&1 | tee ${LOG_FILE}

cd ${OUTPUT_DIR}
python ../cvt_parser.py --logpath ${LOG_FILE} --steps $[${GPUS}*10]
