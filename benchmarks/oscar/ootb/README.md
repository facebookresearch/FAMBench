# Oscar

## Run the benchmark

1. Run container from the docker image.

    - For CUDA: `nvcr.io/nvidia/pytorch:22.02-py3`
    - For ROCm: `rocm/pytorch:rocm5.2_ubuntu20.04_py3.7_pytorch_1.11.0`.

    Sample command:

    ```bash
    docker run -it --ipc=host --privileged --device='/dev/kfd' --device='/dev/dri' --group-add video [-v LOCAL_PATH:CONTAINER_PATH] <IMAGE_NAME> bash
    ```

2. Prepare environment and data.

    Download `azcopy` first from [Download Azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).

    Then run

    ```bash
    bash prep_env_data.sh
    ```

    The `azcopy` step may take a long time (folder size over 80G).

3. Run the benchmark.

    ```bash
    bash run_oscar_train.sh [NODES <1>] [GPUS <8>] [BATCH_SIZE <256>] [EPOCHS <1>]
    ```

## Settings we use for benchmarking MI250X vs. A100

We keep the global batch size for A100 and MI250X the same when running benchmarks and comparing.

For `n`-node run, on each node:

* A100: `MASTER_ADDR=<master ip> NODE_RANK=<node rank> bash run_oscar_train.sh <n> 8 256`
* MI250X: `MASTER_ADDR=<master ip> NODE_RANK=<node rank> bash run_oscar_train.sh <n> 16 128`
