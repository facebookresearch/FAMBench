# BEVT: BERT Pretraining of Video Transformers

References:
- [BEVT on GitHub](https://github.com/xyzforever/BEVT)
- [BEVT paper](https://arxiv.org/abs/2112.01529)

## Run the benchmark

We use a forked BEVT repo here for benchmark purpose. Follow these steps:

1. Build docker image.

   For reproducibility, we provide Dockerfils for running BEVT benchmark on AMD/NVIDIA devices. Please see `Dockerfile.rocm` and `Dockerfile.cuda` for details. User can alternatively use our prebuilt images `mindest/rocm5.2_ubuntu20.04_py3.7_pytorch_1.11.0:bevt`/`mindest/cuda11.3_cudnn8_pytorch_1.11.0:bevt`.

2. Run the docker container.

3. Prepare training data and pretrained checkpoints.

    We will download and prepare these files in path `/tmp/BEVT_DATA`.

    Download DALL-E tokenizer weight:

    ```bash
    bash prep_data.sh
    ```

    Download pretrained checkpoints from [Google drive link 1](https://drive.google.com/file/d/1VHKAH9YA_VD8M8bfGp2Svreqv0iuikB6/view?usp=sharing).

    Download mini-training data from [Google drive link 2](), and extract:

    ```bash
    tar -xvf BEVT_mini_data.tgz
    mv BEVT_mini_data/* /tmp/BEVT_DATA
    ```

4. Run the command.

    Simply run

    ```bash
    bash run_bevt_train.sh [NODES <1>] [GPUS <8>] [BATCH_SIZE <8>] [EPOCHS <1>]
    ```

    And it will run the training and output average iter time (s/it) and throughput (samples/s).

## Sample perf data for MI200 and A100

For A100 we run `bash run_bevt_train.sh 1 a 8`. Global batch size `8a`.

For MI200 we run `bash run_bevt_train.sh 1 b 4`. Global batch size `4b`.

Sample perf data on a single node:

| Device | # GPUs/GCDs | Mini-bsz | Global bsz | Avg iter time (s/it) | Throughput (samples/s) |
| :----: | ----------: | -------: | ---------: |-------------------: | ---------------------: |
|  A100  | 1  | 8 | 8  | 0.4390 | 18.22 |
|  A100  | 2  | 8 | 16 | 0.4515 | 35.44 |
|  A100  | 4  | 8 | 32 | 0.5052 | 63.34 |
|  A100  | 8  | 8 | 64 | 0.7186 | 89.06 |
| MI200  | 2  | 4 | 8  | 0.4705 | 17.00 |
| MI200  | 4  | 4 | 16 | 0.4939 | 32.40 |
| MI200  | 8  | 4 | 32 | 0.5813 | 55.05 |
| MI200  | 16 | 4 | 64 | 0.8583 | 74.57 |
