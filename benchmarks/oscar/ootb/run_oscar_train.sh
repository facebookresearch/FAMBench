export HSA_ENABLE_SDMA=0

NODES=${1:-1}
GPUS=${2:-8}
BS=${3:-256}
EPOCH=${4:-1}

mkdir -p OSCAR_DATA OSCAR_OUTPUT
OUTPUT_DIR=$(realpath OSCAR_OUTPUT)
LOG_FILE=${OUTPUT_DIR}/oscar_${NODES}x${GPUS}.bsz${BS}.log
DATA_DIR=$(realpath OSCAR_DATA)
DATA=${DATA_DIR}/dataset/vqa
MODEL=${DATA_DIR}/best/best

RANK=${RANK:=0}
NODE_COUNT=${NODE_COUNT:=1}
MASTER_ADDR=${MASTER_ADDR:=127.0.0.1}
MASTER_PORT=${MASTER_PORT:=9000}

cd Oscar
export PYTHONPATH=$(pwd):/${PYTHONPATH}
set -ex
python -m torch.distributed.launch \
    --nnodes ${NODE_COUNT} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node=${GPUS} \
    oscar/run_vqa.py \
        -j 4 \
        --img_feature_dim 2054 \
        --max_img_seq_length 50 \
        --data_label_type mask \
        --img_feature_type faster_r-cnn \
        --task_name vqa_text \
        --do_train \
        --do_lower_case \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size 256 \
        --per_gpu_train_batch_size ${BS} \
        --learning_rate 5e-05 \
        --data_dir ${DATA} \
        --model_type bert \
        --model_name_or_path ${MODEL} \
        --output_dir ${OUTPUT_DIR} \
        --label_file ${DATA}/trainval_ans2label.pkl \
        --save_epoch 10 \
        --seed 88 \
        --evaluate_during_training \
        --num_train_epochs ${EPOCH} \
        --logging_steps 40000 \
        --drop_out 0.3 \
        --weight_decay 0.05 \
        --warmup_steps 10 \
        --loss_type bce \
        --img_feat_format pt \
        --classifier linear \
        --cls_hidden_scale 3 \
        --txt_data_dir ${DATA} \
        --fp16 2>&1 | tee ${LOG_FILE}

cd ${OUTPUT_DIR}
python ../oscar_parser.py --logpath ${LOG_FILE} --epochs 0
