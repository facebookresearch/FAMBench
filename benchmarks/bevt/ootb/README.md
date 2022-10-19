# BEVT: BERT Pretraining of Video Transformers

References:
- [BEVT on GitHub](https://github.com/xyzforever/BEVT)
- [BEVT paper](https://arxiv.org/abs/2112.01529)

## Run the benchmark

We use a forked BEVT repo here for benchmark purpose. Follow these steps:

1. Build docker image.

   For reproducibility, we provide Dockerfils for running BEVT benchmark on AMD/NVIDIA devices. Please see `Dockerfile.rocm` and `Dockerfile.cuda` for details. User can alternatively use our prebuilt images `mindest/rocm5.2_ubuntu20.04_py3.7_pytorch_1.11.0:bevt`/`mindest/cuda11.3_cudnn8_pytorch_1.11.0:bevt`.

2. Run the docker container, e.g.,

    ```bash
    sudo docker run --gpus all -it --ipc=host --privileged --device='/dev/kfd' --device='/dev/dri' --group-add video [-v LOCAL_PATH:CONTAINER_PATH] <IMAGE_NAME> bash
    ```

3. Prepare training data and pretrained checkpoints.

    We will download and prepare these files in folder `BEVT_DATA`. Just run

    ```bash
    bash prep_data.sh
    ```

4. Run the benchmark.

    Simply run

    ```bash
    bash run_bevt_train.sh [NODES <1>] [GPUS <8>] [BATCH_SIZE <8>] [EPOCHS <1>] [MAX_STEPS <1200>]
    ```

    And it will run the training and output average iter time (s/it) and throughput (samples/s).

    **Note:** The postprocessing script will skip data of the first 10 timestamps (200 steps) to exclude potential outliers. Therefore by default we will have data for steps 200-1200, which will be used in calculating average iter time.

## Sample perf data for MI250X and A100

For A100, single node, we run `bash run_bevt_train.sh a 8`. Global batch size `8a`.

For MI250X, single node, we run `bash run_bevt_train.sh b 4`. Global batch size `4b`.

Sample perf data on a single node (keeping mini-bsz of MI250X half as that of A100):

| Device | # GPUs/GCDs | Mini-bsz | Global bsz | Avg iter time (s/it) | Throughput (samples/s) |
| :----: | ----------: | -------: | ---------: | -------------------: | ---------------------: |
|  A100  | 1  | 8 | 8  | 0.4390 | 18.22 |
|  A100  | 2  | 8 | 16 | 0.4515 | 35.44 |
|  A100  | 4  | 8 | 32 | 0.5052 | 63.34 |
|  A100  | 8  | 8 | 64 | 0.7186 | 89.06 |
| MI250X | 2  | 4 | 8  | 0.4705 | 17.00 |
| MI250X | 4  | 4 | 16 | 0.4939 | 32.40 |
| MI250X | 8  | 4 | 32 | 0.5813 | 55.05 |
| MI250X | 16 | 4 | 64 | 0.8583 | 74.57 |

## Settings we use for benchmarking MI250X vs. A100

We keep the global batch size for A100 and MI250X the same when running benchmarks and comparing.

For cases with a node, say we use `a` gpus for A100, the running commands are like:

* A100: `bash run_bevt_train.sh <a> 8`
* MI250X: `bash run_bevt_train.sh <a*2> 4`


Run the following command on each node:

* A100: `MASTER_ADDR=<master ip> NODE_COUNT=<node count> RANK=<node rank> bash run_bevt_train.sh 8 8`
* MI250X: `MASTER_ADDR=<master ip> NODE_COUNT=<node count> RANK=<node rank> bash run_bevt_train.sh 16 4`
