from fb5logger import FB5Logger
import time


def dummy_example():
  """Example usage of fb5logger"""

  logger = FB5Logger("example_simple.log") # file to write to
  logger.header("DLRM", "OOTB", "train", "small") # benchmark, implementation, mode, config

  logger.run_start() 
  time.sleep(1) # whatever benchmark here. 
  logger.run_stop(100, 32) # num_batches, batch_size

if __name__ == "__main__":
  dummy_example()