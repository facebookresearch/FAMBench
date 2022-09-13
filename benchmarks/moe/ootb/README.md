# MoE

## Run the benchmark

1. Run container from Docker image.

   - For CUDA: `nvcr.io/nvidia/pytorch:22.02-py3`
   - For ROCm: `rocm/pytorch:rocm5.2_ubuntu20.04_py3.7_pytorch_1.11.0`

    Sample command:

    ```bash
    docker run -it --ipc=host --privileged --device='/dev/kfd' --device='/dev/dri' --group-add video [-v LOCAL_PATH:CONTAINER_PATH] <IMAGE_NAME> bash
    ```

2. Prepare environment and data.

    Simply run

    ```bash
    bash prep_env_data.sh
    ```

3. Run the benchmark.

    ```bash
    bash run_moe_train.sh [NODES <1>] [GPUS <8>] [MAX_TOKENS <16384>]
    ```

    For multi-node case, user will have to trigger the command on all the nodes.

## Settings we use for benchmarking MI250X vs. A100

We keep the global batch size for A100 and MI250X the same when running benchmarks and comparing.

For `n`-node run, on each node:

* A100: `MASTER_ADDR=<master ip> NODE_RANK=<node rank> bash run_moe_train.sh <n> 8 16384`
* MI250X: `MASTER_ADDR=<master ip> NODE_RANK=<node rank> bash run_moe_train.sh <n> 16 8192`
