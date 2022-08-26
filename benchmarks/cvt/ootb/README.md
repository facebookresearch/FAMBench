# CvT (Convolutional vision Transformers)

## Run the benchmark

1. Run container from Docker image.

    - For CUDA: `nvcr.io/nvidia/pytorch:22.02-py3`
    - For ROCm: `rocm/pytorch:rocm5.2_ubuntu20.04_py3.7_pytorch_1.11.0`

    Sample command:

    ```bash
    docker run -it --ipc=host --privileged --device='/dev/kfd' --device='/dev/dri' --group-add video [-v LOCAL_PATH:CONTAINER_PATH] <IMAGE_NAME> bash
    ```

2. Prepare the environment and data.

    Simply run

    ```bash
    bash prep_env_data.sh
    ```

    There might not be enough training data if user is to train on many nodes. User can visit [ImageNet website](https://image-net.org/download.php) to download full size training data (which will be large and take longer time to download), or repeat existing files by simply run `python repeat_fake_data.py [TIMES <5>]`.

3. Run the benchmark.

    ```bash
    bash run_cvt_train.sh [NODES <1>] [GPUS <8>] [BSZ <256>] [RANK <0>] [MASTER_ADDR <127.0.0.1>]
    ```

    Sample command:

    - Training on a single node:
      - On A100: `bash run_cvt_train.sh 1 8`
      - On MI250X: `bash run_cvt_train.sh 1 16 128`
    - Multi-node task: user has to run `bash run_cvt_train.sh <NODES> <GPUS> <BSZ> <RANK> <MASTER_ADDR>` on each node, or use `pdsh` to autorun distributed commands.

    Output log files are automatically parsed to get the overall throughput numbers.

## Settings we use for benchmarking MI250X vs. A100

We keep the global batch size for A100 and MI250X the same when running benchmarks and comparing.

For `n`-node run, on each node:

* A100: `bash run_cvt_train.sh <n> 8 256 <RANK> <MASTER_ADDR>`
* MI250X: `bash run_cvt_train.sh <n> 16 128 <RANK> <MASTER_ADDR>`
