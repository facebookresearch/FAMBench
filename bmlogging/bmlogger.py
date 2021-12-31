import logging
import os
import sys
import json
import time
import loggerconstants as constants

def get_bmlogger(log_file_path = None):
    """
    Get benchmark logger. log_file_path = None returns logger that does nothing.  
    """
    t = Nop() if log_file_path is None else BMLogger
    return t(log_file_path)

class Nop:
    def __init__(self):
        pass

    def __getattr__(self, attr):
        return Nop()

    def __call__(self, *args, **kwargs):
        return Nop()

    def __enter__(self):
        return Nop()

    def __exit__(self):
        pass

    def __repr__(self):
        return "Logger is disabled. bmlogger.get_bmlogger was not passed a file path."

class BMLogger():

    def __init__(self, log_file_path):
        # Create the directory if it doesn't exist.
        log_file_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)

        open(log_file_path, 'w') # create or overwrite file
        self.log_file_path = log_file_path

    def _dump_json(self, d: dict):
        with open(self.log_file_path, 'a') as f:
            json.dump(d, f)
            f.write('\n')

    def _time_ms(self):
        """
        Naive implementation of current time.
        """
        return time.time_ns() * 1e-6

    def log_line(self, log_info : dict, key : str):
        """
        Log a line with a dict of arbitrary form for the data and a string key. 
        """
        log_info['key'] = key
        self._dump_json(log_info)

    def header(self, benchmark_name, implementation_name, mode, config_name, score_metric=constants.EXPS):
        """
        Required for every log. Describes what the benchmark is. 
        """
        header_dict = {
            "benchmark": benchmark_name, 
            "implementation": implementation_name, 
            "mode": mode, 
            "config": config_name,
            "score_metric": score_metric}
        self.log_line(header_dict, constants.HEADER)

    def run_start(self, time_ms = None):
        """
        Records start of logging.
        """
        if(time_ms is None):
            time_ms = self._time_ms()
        start_dict = {"time_ms": time_ms}
        self.log_line(start_dict, constants.RUN_START)

    # TODO: remove batch info args and migrate to record_batch_info
    def run_stop(self, num_batches, batch_size, extra_metadata = None, time_ms = None):
        """
        Records end of logging and any required data. 
        """
        if(time_ms is None):
            time_ms = self._time_ms()
        stop_dict = {"time_ms": time_ms, "num_batches": num_batches, "batch_size": batch_size}
        if extra_metadata is not None:
            stop_dict["extra_metadata"] = extra_metadata
        self.log_line(stop_dict, constants.RUN_STOP)

    def record_batch_info(self, num_batches = None, batch_size = None):
        batch_size_dict = {"batch_size": batch_size}
        self.log_line(batch_size_dict, constants.BATCH_SIZE)
        nbatches_dict = {"num_batches": num_batches}
        self.log_line(nbatches_dict, constants.NUM_BATCHES)
    
    def batch_start(self, time_ms = None):
        """
        Marks beginning of the model processing a batch
        """
        if(time_ms is None):
            time_ms = self._time_ms()
        batch_start_dict = {"time_ms": time_ms}
        self.log_line(batch_start_dict, constants.BATCH_START)

    def batch_stop(self, time_ms = None, batch_size = None):
        """
        Marks end of the model processing a batch
        """
        if(time_ms is None):
            time_ms = self._time_ms()
        batch_stop_dict = {"time_ms": time_ms, "batch_size": batch_size}
        self.log_line(batch_stop_dict, constants.BATCH_STOP)
