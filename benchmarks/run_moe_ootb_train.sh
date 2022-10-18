#!/bin/bash
set -x
pushd moe/ootb
sudo rm -r /MOE_OUTPUT/train_artifacts/checkpoints/*
bash run_moe_train.sh $*
popd
