# Deep Learning Recommendation Model for Personalization and Recommendation Systems:

Copyright (c) Facebook, Inc. and its affiliates.

## Summary
Deep Learning Recommendation Model (DLRM) supports various flags to control the model characteristics and execution sizes. This document introduces a bash script to toggle between the configurations used for benchmarking.

## Getting Started with DLRM
Navigate to the DLRM benchmark script from the top level of the repo.
```
cd benchmarks
```

Run the`run_dlrm_ootb_train.sh` script saving the log files to `results`. By default, this script will run a tiny config. Please make sure the default config is working before proceeding to more complex configs (see Additional Configurations).
```
# This script will save the results to any directory using
# the -l flag. For this example, we use results/.
./run_dlrm_ootb_train.sh -l results
```

Now a log file, `dlrm_ootb_train_tiny.log` will live in `benchmarks/results`. From our current directory, `benchmarks`, run `result_summarizer.py` from the command line.
```
python ../fb5logging/result_summarizer.py -f results
```

The `result_summarizer` will print the summary results to the terminal, which should look like below (with different numbers). Most of the items are self-explanatory, except for the “score” key. Currently this is calculated as examples / second, but our final top-level score metric is subject to change.
```
$ python ../fb5logging/result_summarizer.py -f results/
Summarizing files: ['results/dlrm_ootb_train_tiny.log']
{'benchmark': 'DLRM', 'implementation': 'OOTB', 'mode': 'train', 'config': 'tiny', 'results': {'score': 930.435414697972, 'num_batches': 200, 'batch_size': 64, 'average_batch_time': 0.068785}}
```

### Additional Configurations
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

## Requirements
pytorch-nightly

scikit-learn

numpy

