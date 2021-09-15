'''
Summarizes a set of results.
'''

import json
import os
import re
import sys
import glob


_HEADER_REGEX = r'.*"key": "header".*'
_RUN_START_REGEX = r'.*"key": "run_start".*'
_RUN_STOP_REGEX = r'.*"key": "run_stop".*'

def _find_and_read_row(result : str, regex : str):
    """
    Finds a single row in a log file string and converts it into a dict. 
    """
    row = re.search(regex, result)
    if row is None:
      raise Exception('Failed to match regex!')
    row = json.loads(row.group(0))
    row.pop('key')
    return row

def _calculate_metrics(result : str):
    """
    Calculates runtime in seconds, num_batches
    """
    run_start_row = _find_and_read_row(result, _RUN_START_REGEX)
    run_stop_row = _find_and_read_row(result, _RUN_STOP_REGEX)
    num_batches, batch_size = run_stop_row['num_batches'], run_stop_row['batch_size']

    # calculate runtime
    run_start_time = float(run_start_row['time_ms'])
    run_stop_time = float(run_stop_row['time_ms'])
    seconds_runtime = (run_stop_time - run_start_time) / 1000 

    # calculate throughput, which is score
    throughput = num_batches * batch_size / seconds_runtime # todo if these divisons are by 0
    average_batch_time = seconds_runtime / num_batches
    return {'score' : throughput, 'num_batches' : num_batches, 'batch_size' : batch_size, 'average_batch_time': average_batch_time}

def _create_summary_row(file_path : str):
    """
    Takes a single file path.
    Return JSON row. 
    """
    with open(file_path, 'r') as f:
        result = f.read()

    row = _find_and_read_row(result, _HEADER_REGEX) 
    row['results'] = _calculate_metrics(result)
    return row

def summarize_results(benchmark_folder, csv_file=None):
    """Summarizes a set of results.
    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0, 0.7.0, or 1.0.0.
    """
    rows = []
    pattern = '{folder}/*.log'.format(folder=benchmark_folder)
    result_files = glob.glob(pattern, recursive=True)
    for file_path in result_files:
        row = _create_summary_row(file_path)
        rows.append(row)

    for row in rows:
        print(row)

if __name__ == '__main__':
    summarize_results('.')