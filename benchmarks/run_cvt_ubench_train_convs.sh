steps=100
device='gpu'
forwardonly=0

usage() { echo "Usage: $0 [-s <steps>] [-d <'cpu'|'gpu'>] [-l <dir to save log>]"; exit 1; }

while getopts "s:d:l:fh" flag
do
    case "${flag}" in
        s) steps=${OPTARG};;
        d) device=${OPTARG};;
        l) LOG_DIR=${OPTARG} ;;
        f) forwardonly=1 ;;
        h) usage
    esac
done
shift $((OPTIND-1))

benchmark=cvt
implementation=ubench
mode=train
config=convs
LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${config}.log"

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
echo "Config: ${config}"
echo "Saving FB5 Logger File: ${LOGGER_FILE}"
echo
echo "Running Command:"

if [ $forwardonly -eq 0 ]
then
    (set -x; python cvt/ubench/cvt_ubench_train_convs_driver.py --steps=$steps --warmups 5 --device=gpu --logger_file=${LOGGER_FILE} 2>&1)
else
    (set -x; python cvt/ubench/cvt_ubench_train_convs_driver.py --steps=$steps --warmups 5 --device=gpu --logger_file=${LOGGER_FILE} --forward_only 2>&1)
fi

echo "=== Completed Run ==="
