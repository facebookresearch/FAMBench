#/bin/bash

set -euox pipefail

# Check for two main directory exports required for RNN-T.
if [[ -z "$DATASET_DIR" ]]; then
  echo "ERROR: DATASET_DIR not set! Please set using export DATASET_DIR=\"<path to dataset dir>\"!"
  exit 1
fi
if [[ -z "$RESULT_DIR" ]]; then
  echo "ERROR: RESULT_DIR not set! Please set using export RESULT_DIR=\"<path to result dir>\"!"
  exit 1
fi

# Setting up the conda environment
set +u
source "$($CONDA_EXE info --base)/etc/profile.d/conda.sh"
conda create -n proxy-rnnt python=3.8.3
conda activate proxy-rnnt

# Install PyTorch dependencies
pip install requests bs4 argparse
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install CUDA Toolkit 11.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-11-0

# Switch to CUDA 11.0
if [ ! -d "deps/switch-cuda" ]; then
  git clone https://github.com/phohenecker/switch-cuda.git deps/switch-cuda
fi
source deps/switch-cuda/switch-cuda.sh 11.0
export TORCH_CUDA_ARCH_LIST=8.0

# Install required packages
pip install sox
pip install protobuf==3.11.3
sudo apt-get install libsndfile1 jq numactl cmake sox
pip install unidecode==1.1.1 inflect==4.1.0 pandas==1.1.5 sentencepiece==0.1.94 librosa==0.8.0 soundfile==0.10.3.post1 tensorboard==2.3.0 numba==0.48.0

# Install dllogger and mlcommons logger
pip install https://github.com/NVIDIA/dllogger/archive/26a0f8f1958de2c0c460925ff6102a4d2486d6cc.zip
pip install https://github.com/mlcommons/logging/archive/d08740cadb4188a5ebeb84ad6c68f98c1e129805.zip

# Install Nvidia Dali
pip install --no-cache --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110==0.28.0

# Install warp-rnnt library
pip install warp-rnnt

# Install Nvidia CuDNN
conda install -c nvidia cudnn==8.0.4

# Install apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/8a1ed9e8d35dfad26fb973996319965e4224dcdd.zip

# Other train deps
pip install pyyaml

# Changes to source code, add DATASET_DIR to rnnt/ootb/train/configs/baseline_v3-1024sp.yaml
sed -i 's@sentpiece_model: <your $DATASET_DIR>/sentencepieces/librispeech1023.model@sentpiece_model: '"$DATASET_DIR"'/sentencepieces/librispeech1023.model@' rnnt/ootb/train/configs/baseline_v3-1023sp.yaml

# Install MLPerf loadgen
pushd rnnt/ootb/inference/loadgen
python setup.py install
popd

# Install other inference dependencies
pip install toml==0.10.0
pip install tqdm==4.31.1

# Download the pre-trained model
mkdir -p $RESULT_DIR
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $RESULT_DIR/rnnt.pt

# Download and Extract LibriSpeech Dataset
bash rnnt/ootb/train/scripts/download_librispeech.sh

# Process the .flac files into .wav and .json
bash rnnt/ootb/train/scripts/preprocess_librispeech.sh
