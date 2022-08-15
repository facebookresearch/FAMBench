# Oscar

## Run the benchmark



1. Build docker image.

   - For CUDA: `nvcr.io/nvidia/pytorch:22.02-py3`
   - For ROCm: see `Dockerfile.rocm`.

2. Run the docker container.

3. Prepare environment and data.

    Simply run

    ```bash
    bash prep_env_data.sh
    ```

4. Run the benchmark.

    ```bash
    bash run_oscar_train.sh [NODES <1>] [GPUS <8>] [BATCH_SIZE <128>] [EPOCHS <1>]
    ```
