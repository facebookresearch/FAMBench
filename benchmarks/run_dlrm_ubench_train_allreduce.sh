#!/bin/bash
printUsage() {
  echo
  echo "Usage: $(basename "$0") <options>"
  echo
  echo "Options:"
  echo "  -h                      Prints this help."
  echo "  -l <dir to save log>    Saves FB5 Log to specified directory in first argument."
  echo "  -c <size>               Size in bytes"
  echo
  return 0
}

if [ "$1" == "" ]; then
  printUsage
  exit 0
fi

# Default values
benchmark=dlrm
implementation=ubench
mode=train
collective=allreduce 
size=small

while getopts "hl:c:" flag ;
do
  case "${flag}" in
    h)
      printUsage ; exit 0 ;;
    l)
      LOG_DIR=${OPTARG} ;;
    c)
      size=${OPTARG} ; size_specified=true ;;
  esac
done

size_name=${size}
if [ $size -eq 2200 ]; then
  size_name=small
elif [ $size -eq 9944 ]; then
  size_name=medium
elif [ $size -eq 22372 ]; then
  size_name=large
fi

LOGGER_FILE="${LOG_DIR}/${benchmark}_${implementation}_${mode}_${collective}_${size_name}.log"

get_local_rank=$(echo "python3 -c 'import dlrm.ubench.dlrm_ubench_comms_driver as comms; print(comms.get_local_rank())'")
rank=$(eval $get_local_rank | head -n 1)
master_ip="${5:-localhost}"

if [ $rank -eq 0 ]; then
  echo "=== Launching FB5 ==="
  echo "Benchmark: ${benchmark}"
  echo "Implementation: ${implementation}"
  echo "Mode: ${mode}"
  echo "Collective: ${collective}"
  echo "Size: ${size}"
  echo "Saving FB5 Logger File: ${LOGGER_FILE}"
  echo "Running Command:"
fi

(set -x; python3 "${benchmark}/${implementation}/dlrm_ubench_comms_driver.py" --master_ip=${master_ip} --fb5logger=${LOGGER_FILE} --collective=all_reduce --size=${size} 2>&1)

if [ $rank -eq 0 ]; then
  echo "=== Completed Run ==="
fi
