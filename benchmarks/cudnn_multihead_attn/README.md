# Multi-head Attention CuDNN Benchmark

## Dependencies

This benchmark has been tested with the following dependencies:

CUDA 11.5 / 11.6

CuDNN 8.3.1 / 8.6.0

Sqlite3

gflags

Anaconda

PyTorch 1.10 / 1.12

## Ubuntu 20.04 Env Setup

(Tested on an AWS EC2 p3.16xlarge instance and p4d.24xlarge instance running Ubuntu Server 20.04 LTS instance)

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
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb ;
sudo dpkg -i cuda-keyring_1.0-1_all.deb ;
sudo apt-get update ;
sudo apt-get install libcudnn8 ;
sudo apt-get install libcudnn8-dev ;
```

Install gflags and sqlite3.

```
sudo apt-get install -y libgflags-dev libsqlite3-dev
```

Add CUDA NVCC path and library paths to $PATH and $LD_LIBRARY_PATH respectively.

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

Create & activate a conda environment.

```
conda create --name my_env python=3.8 -y
source activate my_env
```

Install PyTorch 1.10.

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Clone FAMBench. Save the root directory path in a variable for convenience.

```
git clone https://github.com/facebookresearch/FAMBench.git
export FAMBENCH=$(pwd)/FAMBench
```

Compile cudnn_multihead_attn_benchmark.cu.

```
cd $FAMBENCH/benchmarks/cudnn_multihead_attn/
nvcc -o mha -I/usr/include -L/usr/lib/x86_64-linux-gnu -l:libcudnn.so.8.6.0 -l:libsqlite3.so.0 -l:libgflags.so -lpthread -ldl --std=c++17 cudnn_multihead_attn_benchmark.cu
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
