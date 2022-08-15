# /bin/bash

set -ex

NODES=${1:-1}
GPUS=${2:-8}
BATCH_SIZE=${3:-128}
EPOCH=${4:-1}
LOGPATH=${WORKSPACE}/oscar_${GPUS}gpus.log

export NODE_COUNT=${NODES}

# init environment
bash mount_data.sh

# run model
bash run_model.sh ${GPUS} ${BATCH_SIZE} ${EPOCH} ${LOGPATH}

python oscar_parser.py --logpath ${LOGPATH}  --epochs 0