# Deep Learning Recommendation Model for Personalization and Recommendation Systems:

Copyright (c) Facebook, Inc. and its affiliates.

## Summary
Deep Learning Recommendation Model (DLRM) supports various flags to control the model characteristics and execution sizes. This document introduces a bash script to toggle between the configurations used for benchmarking.

## Getting Started with DLRM
Here is an example initial run. Run the following commands in terminal.

Starting from the top level of the repo,
```
cd benchmarks
```
Now we are at proxyworkloads/benchmarks

Run one of the DLRM benchmarks. This script will log to the 
directory using the -l flag. Here, log to results/.
```
./run_dlrm_ootb_train.sh -l results
```

Create summary table and save to results/summary.txt
```
python ../fb5logging/result_summarizer.py -f results 
```

See and/or run proxyworkloads/benchmarks/run_all.sh for a runnable example. Please note that to run it, your current dir must be at proxyworkloads/benchmarks.

### Additional DLRM Configurations
You may choose to run your own model configuration. To do so, create a config file containing all flags for `dlrm_s_pytorch.py` on a single line. For example, create a file called `dlrm_tutorial` with contents:

```
--mini-batch-size=64 --test-mini-batch-size=64 --test-num-workers=0 --num-batches=1000 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time
```

Run the `run_dlrm_ootb_train.sh` script with the `-c` flag to specify which config file to use:

```
./run_dlrm_ootb_train.sh -l results -c dlrm_tutorial
```

In this example, you should see an output similar to this:

```
$ ./run_dlrm_ootb_train.sh -l results -c dlrm_tutorial
=== Launching FB5 ===
Benchmark: dlrm
Implementation: ootb
Mode: train
Config: dlrm_tutorial
Saving FB5 Logger File: results/dlrm_ootb_train_dlrm_tutorial.log

Running Command:
+ python dlrm/ootb/dlrm_s_pytorch.py --mini-batch-size=64 --test-mini-batch-size=64 --test-num-workers=0 --num-batches=1000 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time
world size: 1, current rank: 0, local rank: 0
Using CPU...
time/loss/accuracy (if enabled):
Finished training it 100/1000 of epoch 0, 56.60 ms/it, loss 0.084849
Finished training it 200/1000 of epoch 0, 44.95 ms/it, loss 0.082306
Finished training it 300/1000 of epoch 0, 45.26 ms/it, loss 0.083103
Finished training it 400/1000 of epoch 0, 47.32 ms/it, loss 0.080760
Finished training it 500/1000 of epoch 0, 46.90 ms/it, loss 0.084727
Finished training it 600/1000 of epoch 0, 45.55 ms/it, loss 0.083395
Finished training it 700/1000 of epoch 0, 47.67 ms/it, loss 0.084470
Finished training it 800/1000 of epoch 0, 44.90 ms/it, loss 0.083775
Finished training it 900/1000 of epoch 0, 46.24 ms/it, loss 0.082480
Finished training it 1000/1000 of epoch 0, 46.44 ms/it, loss 0.082861
=== Completed Run ===
```

### Inference
You may also choose to run DLRM in Inference mode. To do so, follow the same steps as above using the `run_dlrm_ootb_infer.sh` script instead.

## Requirements
pytorch-nightly

scikit-learn

numpy

## Optional
### fbgemm_gpu
Install additional requirements:
```
conda install jinja2
conda install nvidiacub
```
Set export paths:
```
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUB_DIR=${CUB_DIR}
```
Clone repo:
```
git clone https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
git submodule sync
git submodule update --init --recursive    
```
Run installer:
```
python setup.py build develop
```
Copy shared object file
```
cp fbgemm_gpu_py.so /<proxyworkloads root>/benchmarks
```
Enable fbgemm_gpu by adding command line argument: --use-fbgemm-gpu
