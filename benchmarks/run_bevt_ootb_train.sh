NODES=1
NUM_GPUS=8
BS=8
EP=8

usage() { echo "Usage: $0 [-n <NODES>] [-g <NUM_GPUS>] [-b <BATCH_SIZE>] [-e <EPOCHS>]"; exit 1; }

while getopts "n:g:b:e:h" flag
do
    case "${flag}" in
        n) NODES=${OPTARG} ;;
        g) NUM_GPUS=${OPTARG} ;;
        b) BS=${OPTARG} ;;
        e) EP=${OPTARG} ;;
        h) usage
    esac
done
shift $((OPTIND-1))

benchmark=bevt
implementation=ootb
mode=train

echo "=== Launching FB5 ==="
echo "Benchmark: ${benchmark}"
echo "Implementation: ${implementation}"
echo "Mode: ${mode}"
echo
echo "Running Command:"

cd bevt/ootb
bash prep_env_data.sh
cd BEVT
bash run_bevt_train.sh ${NODES} ${NUM_GPUS} ${BS} ${EP}

echo "=== Completed Run ==="
