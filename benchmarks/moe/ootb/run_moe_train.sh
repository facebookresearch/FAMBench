
NODES=${1:-1}
GPUS=${2:-8}
MAX_TOKENS=${3:-8192}

export MAX_TOKENS=${MAX_TOKENS}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $[${GPUS}-1])
export DATABIN=$(realpath wmt16_en_de/databin)

OUTPUT_DIR=MOE_OUTPUT
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/moe_${NODES}x${GPUS}.log

export WORKSPACE=${OUTPUT_DIR}

export NUM_GPUS=8
export EP_WORLD_SIZE=8
export NUM_EXPERTS=8
export NODE_COUNT=${NODES}
unset WORLD_SIZE

cd fairseq/examples/deepspeed/moe_e/

if [ ${NODES} -eq 1 ]; then
    run_script=run.sh
else
    run_script=run-distributed.sh
fi

# Shorten the training loop
sed -i "s/max-update 300000/max-update 200/g" ${run_script}
sed -i "s/validate-interval-updates 20/validate-interval-updates 1000/g" ${run_script}

bash ${run_script} 2>&1 | tee ${LOG_FILE}

python moe_parser.py --logpath ${LOG_FILE} --select_epoch 0
