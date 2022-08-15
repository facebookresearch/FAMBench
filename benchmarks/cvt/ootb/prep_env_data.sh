#!/bin/bash
set -ex

cd CvT
pip install -r requirements-alt.txt

mkdir -p ../DATASET
cd ..

if [ ! -d DATASET/imagenet ] ; then
    if [ ! -f imagenette2-320.tgz ] ; then
        wget -q https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
    fi
    tar xf imagenette2-320.tgz -C DATASET
    mv DATASET/imagenette2-320 DATASET/imagenet
fi
