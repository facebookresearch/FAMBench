"""
Master list of constants for logger
Mostly logger keys, but some other constants as well.
"""

# loggerkey - header
HEADER = "header"

# loggerkey - timing info
EPOCH_START = "epoch_start"
EPOCH_STOP = "epoch_stop"
RUN_START = "run_start"
RUN_STOP = "run_stop"

# loggerkey - run information
NUM_BATCHES = "num_batches"
BATCH_SIZE = "batch_size"
FLOPS = "flops"

# loggerkey - model hyperparameters
LEARNING_RATE = "learning_rate"

# type of summary view saved to file
INTERMEDIATE_VIEW_MAXTHROUGHPUT = "intermediate_view_maxthroughput" # table view where duplicates prune down to max throughput within batch latency requirement
INTERMEDIATE_VIEW = "intermediate_view" # table view with duplicates
RAW_VIEW = "raw_view" # json view

# available types of score metrics
EXPS = "exps" # examples/sec (throughput)
TFPS = "tfps" # teraflops/sec (floating point ops rate)
GBPS = "gbps" # gb/sec 
