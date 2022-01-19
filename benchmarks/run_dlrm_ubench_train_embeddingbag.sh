steps=100
device='cpu'
dataset='small'
dtype='float'
forwardonly=0

usage() { echo "Usage: $0 [-s <steps>] [-d <'cpu'|'gpu'>] [-l <dir to save log>] [-c <config>] [-t <'float'|'float16'|'bfloat16']"; exit 1; }

while getopts "s:d:l:c:t:fh" flag
do
    case "${flag}" in
        s) steps=${OPTARG};;
        d) device=${OPTARG};;
        l) LOG_DIR=${OPTARG} ;;
        c) dataset=${OPTARG} ;;
        t) dtype=${OPTARG} ;;
        f) forwardonly=1 ;;
        h) usage
    esac
done
shift $((OPTIND-1))

benchmark=dlrm
implementation=ubench
mode=train
config=embeddingbag_${dataset}_${dtype}
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
echo "Config: ${config}"
echo "Saving FB5 Logger File: ${LOGGER_FILE}"
echo
echo "Running Command:"

if [ "$device" == "gpu" ]
then
  if [ $forwardonly -eq 0 ]
  then
     (set -x; python dlrm/ubench/dlrm_ubench_train_embeddingbag_driver.py --steps=$steps --device=gpu --fb5logger=${LOGGER_FILE} -d "${dataset}" -t ${dtype} 2>&1)
  else
     (set -x; python dlrm/ubench/dlrm_ubench_train_embeddingbag_driver.py --steps=$steps --device=gpu --fb5logger=${LOGGER_FILE} -d "${dataset}" -t ${dtype} --forward_only 2>&1)
  fi
else
  (set -x; python dlrm/ubench/dlrm_ubench_train_driver.py --steps=$steps --device=$device --fb5logger=${LOGGER_FILE} emb --dataset="${dataset}" 2>&1)
fi

echo "=== Completed Run ==="
