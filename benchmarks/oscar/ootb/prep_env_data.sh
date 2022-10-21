#!?usr/bin/env bash

set -ex

# Install dependencies
cd Oscar
pip install -r requirements.txt

# Download data
cd -
mkdir -p OSCAR_DATA
cd OSCAR_DATA
# Best checkpoint (path OSCAR_DATA/best/best)
wget -q https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/vqa/base/best.zip
unzip best.zip
rm best.zip
# Training data (path OSCAR_DATA/dataset/vqa)
# Download via azcopy
wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1
./azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/datasets/vqa dataset --recursive
