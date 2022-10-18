#!/bin/bash
set -x
pushd oscar/ootb
bash run_oscar_train.sh $*
popd
