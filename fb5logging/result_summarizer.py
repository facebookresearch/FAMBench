'''
Summarizes a set of results.
'''

import argparse
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
      raise Exception('Failed to match regex: '.format(regex))
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
    run_start_row = _find_and_read_row(result, _RUN_START_REGEX)
    run_stop_row = _find_and_read_row(result, _RUN_STOP_REGEX)
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

def _flatten_dict(d: dict):
    """
    Flattens a nested dictionary, not in-place
    """
    res = {}
    for key, val in d.items():
        if isinstance(val, dict):
            res.update(_flatten_dict(val))
        else:
            res[key] = val
    return res

def _create_summary_row(file_path : str):
    """
    Takes a single file path.
    Return JSON row.
    """
    with open(file_path, 'r') as f:
        result = f.read()

    row = _find_and_read_row(result, _HEADER_REGEX)
    results = _calculate_metrics(result)
    row['results'] = results
    return row

def _lst_to_file(lst: list, file_path):
    for i in range(len(lst)):
        lst[i] = str(lst[i])
    delimiter = ' ' #space delimiter
    with open(file_path, 'a') as f:
        f.write(delimiter.join(lst) + '\n')

def _rows_to_file(rows: list[dict], folder_path: str):
    """
    Save list of summary rows into a human-readable table in a file
    """
    file_path = folder_path + '/summary.txt'

    if(len(rows) == 0):
        return

    all_keys = _flatten_dict(rows[0]).keys()
    top_level_keys = [
        "benchmark",
        "implementation",
        "mode",
        "config",
        "score"]
    other_keys = [k for k in all_keys if k not in top_level_keys]
    keys_in_order = top_level_keys + other_keys
    _lst_to_file(keys_in_order, file_path)
    for row in rows:
        flattened_row = _flatten_dict(row)
        top_val_lst = [flattened_row[k] for k in top_level_keys]
        other_val_lst = [flattened_row[k] for k in other_keys]
        combined_lst = top_val_lst + other_val_lst
        _lst_to_file(combined_lst, file_path)
        

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
<<<<<<< HEAD

    return rows
=======
>>>>>>> 71dd7c28769faba599c19746d8b40d9c8d045833

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a folder of logged benchmark result files."
    )
    parser.add_argument('-f', '--benchmark-folder', type=str, default='.')
    return parser

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    rows = summarize_results(args.benchmark_folder)
    _rows_to_file(rows, args.benchmark_folder)
