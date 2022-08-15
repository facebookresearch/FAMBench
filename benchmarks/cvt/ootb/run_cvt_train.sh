NODES=${1:-1}
GPUS=${2:-8}

NODE_RANK=${3:-0}
MASTER_ADDR=${4:-127.0.0.1}

export NODE_COUNT=${NODES}
export RANK=${NODE_RANK}
export MASTER_ADDR=${MASTER_ADDR}

OUTPUT_DIR=../OUTPUT/
LOG_FILE=${OUTPUT_DIR}/cvt_${NODES}x${GPUS}.log

set -ex

cd CvT
bash run-alt.sh -g ${GPUS} -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml \
    DATASET.ROOT ../DATASET/imagenet OUTPUT_DIR ${OUTPUT_DIR} | tee ${LOG_FILE}

python cvt_parser.py --logpath ${LOG_FILE} --steps $[${GPUS}*10]
