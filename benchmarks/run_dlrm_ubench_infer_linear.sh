steps=1000
device='cpu'
dataset='A'
dtype='float'
TRT=0
MIGRAPHX=0

usage() { echo "Usage: $0 [-s <steps>] [-d <'cpu'|'gpu'>] [-l <dir to save log>] [-c <config>] [-t <'float'|'float16'|'bfloat16'] [-T] [-M]"; exit 1; }

while getopts "s:d:l:c:t:hTM" flag
do
    case "${flag}" in
        s) steps=${OPTARG};;
        d) device=${OPTARG};;
        l) LOG_DIR=${OPTARG} ;;
        c) dataset=${OPTARG} ;;
        t) dtype=${OPTARG} ;;
        h) usage ;;
        T) TRT=1 ;;
        M) MIGRAPHX=1 ;;
    esac
done
shift $((OPTIND-1))

benchmark=dlrm
implementation=ubench
mode=infer
config=linear_${dataset}_${dtype}
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
echo "Config: ${config}"
echo "Saving FB5 Logger File: ${LOGGER_FILE}"
echo
echo "Running Command:"

if [ $TRT -eq 1 ]
then
    (set -x; python dlrm/ubench/dlrm_ubench_infer_driver.py --steps=$steps --warmups 1000 --device=$device --fb5logger=${LOGGER_FILE} linear --dataset="${dataset}" --dtype="${dtype}" --use-trt 2>&1)
elif [ $MIGRAPHX -eq 1 ]
then
    (set -x; python dlrm/ubench/dlrm_ubench_infer_driver.py --steps=$steps --warmups 1000 --device=$device --fb5logger=${LOGGER_FILE} linear --dataset="${dataset}" --dtype="${dtype}" --use-migraphx 2>&1)
else
    (set -x; python dlrm/ubench/dlrm_ubench_infer_driver.py --steps=$steps --warmups 1000 --device=$device --fb5logger=${LOGGER_FILE} linear --dataset="${dataset}" --dtype="${dtype}" 2>&1)
fi

echo "=== Completed Run ==="
