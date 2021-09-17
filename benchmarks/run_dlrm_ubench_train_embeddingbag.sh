steps=100
device='cpu'
dataset='A'

usage() { echo "Usage: $0 [-s <steps>] [-d <'cpu'|'gpu'>] [-l <dir to save log>]"; exit 1; }

while getopts "s:d:l:h" flag
do
    case "${flag}" in
        s) steps=${OPTARG};;
        d) device=${OPTARG};;
        l) LOG_DIR=${OPTARG} ;;
        h) usage
    esac
done
shift $((OPTIND-1))

benchmark=dlrm
implementation=ubench
mode=train
config=embeddingbag_small
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
echo "Config: ${config}"
echo "Saving FB5 Logger File: ${LOGGER_FILE}"
echo
echo "Running Command:"

(set -x; python dlrm/ubench/dlrm_ubench_train_driver.py --steps=$steps --device=$device --fb5logger=${LOGGER_FILE} emb --dataset='small' 2>&1)

echo "=== Completed Run ==="
