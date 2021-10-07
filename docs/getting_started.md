## Getting Started: DLRM Example
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