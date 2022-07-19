#/bin/bash
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license

set -ex

# Update CMake
apt update && \
    apt install -y software-properties-common lsb-release && \
    apt clean all && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt list -a cmake && \
    apt update && apt install -y cmake

# Install Deps
apt install -y sox jq ninja-build libomp5

# Get torchaudio #TODO: change when upstreamed.
git clone --recurse-submodules -b rocm_rnnt https://github.com/ROCmSoftwarePlatform/audio.git
cd audio/ && python3 setup.py install && cd -

# Install train requirements.
pip install -r ./benchmarks/rnnt/ootb/train/requirements.txt

# Set data location.
mkdir -p $DATASET_DIR

# Fix benchmarks/rnnt/ootb/train/configs/baseline_v3-1023sp.yaml
sed -i "s@<your $DATASET_DIR>@$DATASET_DIR@" ./benchmarks/rnnt/ootb/train/configs/baseline_v3-1023sp.yaml

if [ -z "$(ls -A ${DATASET_DIR})" ]; then
    # Download and preprocess
    bash ./benchmarks/rnnt/ootb/train/scripts/download_librispeech.sh
    bash ./benchmarks/rnnt/ootb/train/scripts/preprocess_librispeech.sh
fi
