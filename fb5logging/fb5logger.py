import logging
import os
import sys
import json
import time

class FB5Logger():

    def __init__(self, log_file_path):
        # Create the directory if it doesn't exist.
        log_file_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)

        open(log_file_path, 'w') # create or overwrite file
        self.log_file_path = log_file_path

    def _dump_json(self, dict):
        with open(self.log_file_path, 'a') as f:
            json.dump(dict, f)
            f.write('\n')

    def _time_ms(self):
        """
        Naive implementation of current time.
        """
        return round(time.time() * 1000)

    def header(self, benchmark_name, implementation_name, mode, config_name):
        header_dict = {"benchmark": benchmark_name, "implementation": implementation_name, "mode": mode, "config": config_name, "key": "header"}
        self._dump_json(header_dict)

    def run_start(self, time_ms = None):
        """
        Records start of logging.
        """
        if(time_ms is None):
            time_ms = self._time_ms()
        start_dict = {"time_ms": time_ms, "key": "run_start"}
        self._dump_json(start_dict)

    def run_stop(self, num_batches, batch_size, extra_metadata = None, time_ms = None):
        """
        Records end of logging and any required data. 
        """
        if(time_ms is None):
            time_ms = self._time_ms()
        start_dict = {"time_ms": self._time_ms(), "num_batches": num_batches, "batch_size": batch_size, "key": "run_stop"}
        if extra_metadata is not None:
            start_dict["extra_metadata"] = extra_metadata
        self._dump_json(start_dict)

