steps=100
device='cpu'
dataset='A'

usage() { echo "Usage: $0 [-s <steps>] [-d <'cpu'|'gpu'>]"; exit 1; }

while getopts "s:d:" flag
do
    case "${flag}" in
        s) steps=${OPTARG};;
        d) device=${OPTARG};;
    esac
done
shift $((OPTIND-1))

python dlrm_ubench_train_driver.py --steps=$steps --device=$device linear --dataset='A'
