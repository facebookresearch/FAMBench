NUM_GPUS=8
task=train
LOG_DIR=.

usage() { echo "Usage: $0 [-g <NUM_GPUS>] [-t <'train'|'test'>]"; exit 1; }

while getopts "g:t:h" flag
do
    case "${flag}" in
        g) NUM_GPUS=${OPTARG} ;;
        t) task=${OPTARG} ;;
        h) usage
    esac
done
shift $((OPTIND-1))

benchmark=cvt
implementation=ootb
mode=${task}
# LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_g${NUM_GPUS}.log"

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
# echo "Saving FB5 Logger File: ${LOGGER_FILE}"
echo
echo "Running Command:"

(
    set -x;
    cd cvt/ootb;
    bash prep_env_data.sh;
    cd CvT;
    bash run-alt.sh -g ${NUM_GPUS} -t ${task} --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml \
        DATASET.ROOT /tmp/DATASET/imagenet OUTPUT_DIR /tmp/OUTPUT
)

echo "=== Completed Run ==="
