# CvT (Convolutional vision Transformers)

## Run the benchmark

1. Run container from Docker image.

   - For CUDA: `nvcr.io/nvidia/pytorch:22.02-py3`
   - For ROCm: `rocm/pytorch:rocm5.2_ubuntu20.04_py3.7_pytorch_1.11.0`

2. Prepare the environment and data.

    Simply run

    ```bash
    bash prep_env_data.sh
    ```

3. Run the benchmark.

    ```bash
    bash run_cvt_train.sh [NODES <1>] [GPUS <8>] [RANK <0>] [MASTER_ADDR <127.0.0.1>]
    ```

    Sample command:

    - Training on a single node: `bash run_cvt_train.sh 1 8`
    - Multi-node task: user has to run `bash run_cvt_train.sh <NODES> <GPUS> <RANK> <MASTER_ADDR>` on each node, or use `pdsh` to autorun distributed commands.

    Output files are automatically parsed to get the overall throughput numbers.
