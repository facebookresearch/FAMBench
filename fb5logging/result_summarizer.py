'''
Summarizes a set of results.
'''

import argparse
import json
import os
import re
import sys
import glob
import loggerconstants as constants

def _find_and_read_row(result : str, key : str):
    """
    Finds a single row in a log file string and converts it into a dict.
    """
    regex = r'.*"key": "{}".*'.format(key)
    row = re.search(regex, result)
    if row is None:
      raise Exception('Failed to match regex!')
    row = json.loads(row.group(0))
    row.pop('key')
    return row

def _calculate_metrics(result : str):
    """
    Calculates results dictionary with the following keys:
      score = examples / sec
      num_batches = number of batches
      batch_size = number of examples per batch
      average_batch_time = seconds / batch 
      extra_metadata = any extra information - may be used for future summary info
    """
    run_start_row = _find_and_read_row(result, constants.RUN_START)
    run_stop_row = _find_and_read_row(result, constants.RUN_STOP)
    num_batches, batch_size = run_stop_row['num_batches'], run_stop_row['batch_size']

    # calculate runtime
    run_start_time = float(run_start_row['time_ms'])
    run_stop_time = float(run_stop_row['time_ms'])
    seconds_runtime = (run_stop_time - run_start_time) / 1000

    # calculate throughput, which is score
    throughput = num_batches * batch_size / seconds_runtime # TODO if these divisons are by 0 catch exception
    average_batch_time = seconds_runtime / num_batches
    result = {'score' : throughput, 'num_batches' : num_batches, 'batch_size' : batch_size, 'average_batch_time': average_batch_time}

    #append extra_metadata if present
    if 'extra_metadata' in run_stop_row:
        result['extra_metadata'] = run_stop_row['extra_metadata']
    return result

def _create_summary_row(file_path : str):
    """
    Takes a single file path.
    Return JSON row.
    """
    with open(file_path, 'r') as f:
        result = f.read()

    row = _find_and_read_row(result, constants.HEADER)
    results = _calculate_metrics(result)
    row['results'] = results
    return row

def summarize_results(benchmark_folder):
    """
    Summarizes a set of results.
    """
    rows = []
    pattern = '{folder}/*.log'.format(folder=benchmark_folder) # TODO allow other kinds of files
    result_files = glob.glob(pattern, recursive=True)
    if(len(result_files) == 0):
        print('No result files to summarize!')
        return
    print('Summarizing files: {}'.format(result_files))
    for file_path in result_files:
        row = _create_summary_row(file_path)
        rows.append(row)

    for row in rows:
        print(row) 

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a folder of logged benchmark result files."
    )
    parser.add_argument('-f', '--benchmark-folder', type=str, default='.')
    return parser

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    summarize_results(args.benchmark_folder)
