from fb5logger import get_fb5logger
import time


def dummy_example():
  """Example usage of fb5logger"""

  logger = get_fb5logger("results/example_simple.log") # file to write to. works only with .log
  logger.header("DLRM", "OOTB", "train", "small") # benchmark, implementation, mode, config

  logger.run_start()
  time.sleep(1) # whatever benchmark here.
  logger.run_stop(100, 32) # num_batches, batch_size

if __name__ == "__main__":
  dummy_example()
