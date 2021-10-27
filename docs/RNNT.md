# RNN-T from MLCommons w/ LibriSpeech Dataset

## Requirements

 - CUDA 11.0
 - Microconda
 - GPU device compatible with CUDA 11.0
 - sox
 - libsndfile1

## Training

This document provides the detailed instructions to start training RNN-T models with Open-Source LibriSpeech Dataset. The repository can be found here: https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch

### Setting up the conda environment
```
conda create -n py383 python=3.8.3
conda activate py383
pip install requests bs4 argparse
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

### Switch to CUDA 11.0
```
git clone https://github.com/phohenecker/switch-cuda.git
source ~/cluster/work/switch-cuda/switch-cuda.sh 11.0
export TORCH_CUDA_ARCH_LIST=8.0
```

### Getting LibriSpeech ready
```
# Install required packages
sudo apt-get install sox libsndfile1 jq numactl git cmake
pip install unidecode==1.1.1 inflect==4.1.0 pandas==1.1.5 sentencepiece==0.1.94 librosa==0.8.0 soundfile==0.10.3.post1 tensorboard==2.3.0

# Set-up directories and exports
# Pick a mounted location that can hold up to 500GB of dataset data
export DATASET_DIR=<your path>/rnnt/datasets
export RESULT_DIR=<your path>/rnnt/results

# Download and Extract LibriSpeech Dataset
./download_librispeech.sh

# Process the .flac files into .wav and .json
./preprocess_librispeech.sh
```

### Getting Training running:
```
# Install dllogger and mlcommons logger
pip install https://github.com/NVIDIA/dllogger/archive/26a0f8f1958de2c0c460925ff6102a4d2486d6cc.zip
pip install https://github.com/mlcommons/logging/archive/d08740cadb4188a5ebeb84ad6c68f98c1e129805.zip

# Install Nvidia Dali
pip install --no-cache --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110==0.28.0

# Install Warp-Transducer library
git clone https://github.com/HawkAaron/warp-transducer deps/warp-transducer
cd deps/warp-transducer/
git checkout f546575109111c455354861a0567c8aa794208a2
sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/g' CMakeLists.txt
sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")/g' CMakeLists.txt
mkdir build
cd build/
cmake ..
make -j32
export WARP_RNNT_PATH=`pwd`
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
cd ../pytorch_binding
python3 setup.py install
rm -rf ../tests test ../tensorflow_binding
cd ../../..

# Install Nvidia CuDNN
pip install -c nvidia cudnn==8.0.4

# Install apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/8a1ed9e8d35dfad26fb973996319965e4224dcdd.zip

# Other deps
pip install pyyaml
```

### Changes to Source Code:

* Change configs/baseline_v3-1024sp.yaml’s tokenizer: sentpiece_model: to your $DATASET_DIR/sentencepieces/librispeech1023.model

### Finally train with command:
```
bash rnnt/ootb/train/scripts/train.sh $DATASET_DIR/LibriSpeech rnnt/ootb/train/configs/baseline_v3-1023sp.yaml $RESULT_DIR
```
At this point, you should be able to see training epochs.

## Inference

This document provides the detailed instructions to run inference on a pre-trained model from MLCommons against the Open-Source LibriSpeech dataset. The repository can be found here: https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt

### Installing dependencies

Using the same conda environment as training:
```
conda activate py383
```

Install MLPerf loadgen and additional packages:
```
# Install MLPerf loadgen
pushd inference/loadgen
python setup.py install
popd

# Install dependencies
pip install toml==0.10.0
pip install tqdm==4.31.1
```

Download the pre-trained model:
```
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $RESULT_DIR/rnnt.pt
```

### Finally run inference with the command:
```
python rnnt/ootb/inference/run.py --backend pytorch --dataset_dir $DATASET_DIR/LibriSpeech --manifest $DATASET_DIR/LibriSpeech/librispeech-dev-clean-wav.json --pytorch_config_toml rnnt/ootb/inference/pytorch/configs/rnnt.toml --pytorch_checkpoint $RESULT_DIR/rnnt.pt --scenario Offline --log_dir $RESULT_DIR/Offline_pytorch_rerun
```
At this point, wait for inference to finish and produce results.
