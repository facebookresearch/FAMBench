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
