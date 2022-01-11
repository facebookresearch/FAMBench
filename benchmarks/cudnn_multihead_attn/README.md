# Multi-head Attention CuDNN Benchmark

## Dependencies

This benchmark has been tested with the following dependencies:

CUDA 11.5

CuDNN 8.3.1

Sqlite3

gflags

Anaconda

PyTorch 1.10

## Ubuntu 20.04 Env Setup

(Tested using an AWS EC2 p3.16xlarge Ubuntu Server 20.04 LTS instance)

Install build essentials.

```
sudo apt-get update
sudo apt install build-essential
```

Install CUDA.

```
wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
sudo sh cuda_11.5.1_495.29.05_linux.run
```

Install CuDNN.

(The following steps are based on https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.3.1.*-1+cuda11.5
sudo apt-get install libcudnn8-dev=8.3.1.*-1+cuda11.5
```

Install gflags and sqlite3.

```
sudo apt-get install -y libgflags-dev libsqlite3-dev
```

Add CUDA NVCC and library paths to $PATH and $LD_LIBRARY_PATH respectively.

```
echo -e "export PATH=\"/usr/local/cuda/bin:\$PATH\"\nexport LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:\$LD_LIBRARY_PATH\"\n$(cat ~/.bashrc)" > ~/.bashrc
source ~/.bashrc
```

Install Anaconda.

```
curl https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh --output Anaconda3-5.3.0-Linux-x86_64.sh
sh Anaconda3-5.3.0-Linux-x86_64.sh
export PATH="/home/${USER}/anaconda3/bin:$PATH"
```

Create & activate an Anaconda environment.

```
conda create --name my_env python=3.8 -y
source activate my_env
```

Install PyTorch 1.10.

```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

```

Clone FAMBench. Save the root directory in a variable for convenience.

```
git clone https://github.com/facebookresearch/FAMBench.git
export FAMBENCH=$(pwd)/FAMBench
```

Compile cudnn_multihead_attn_benchmark.cu.

```
cd $FAMBENCH/benchmarks/
nvcc -o cudnn_multihead_attn/mha -I/usr/include -L/usr/lib/x86_64-linux-gnu -l:libcudnn.so.8.3.1 -l:libsqlite3.so.0 -l:libgflags.so -lpthread -ldl --std=c++17 cudnn_multihead_attn/cudnn_multihead_attn_benchmark.cu
```

## Run the benchmarks

### Run the FAMBench-preselected benchmarks

```
cd $FAMBENCH/benchmarks/
./cudnn_multihead_attn_benchmarks.sh
```

### Run a custom benchmark

```
cd $FAMBENCH/benchmarks/cudnn_multihead_attn
python multihead_attn_make_ref.py --emb-dim=<number> --num-heads=<number> --seq-len=<number> --batch-size=<number> --double-precision-ref-data
./mha --training=true --iterations=100 --double_precision=true --debug_mode=true --print_accuracy_stats=true
```
