'''
Summarizes a set of results.
'''

import argparse
import json
import os
import re
import sys
import glob
import math
import loggerconstants as constants

## Utility
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

def _dump_json(d: dict, file_path: str):
    with open(file_path, 'a') as f:
        json.dump(d, f)
        f.write('\n')

def _lst_to_file(lst: list, file_path: str):
    for i in range(len(lst)):
        lst[i] = str(lst[i])
    delimiter = ' ' #space delimiter
    with open(file_path, 'a') as f:
        f.write(delimiter.join(lst) + '\n')

def _find_and_read_row_multiple(log_str : str, key : str):
    """
    Finds multiple rows in a log file string and converts it into list of dicts.
    Gives in order of how it appears in the document.
    """
    regex = r'.*"key": "{}".*'.format(key)
    row_lst = re.findall(regex, log_str)
    for i, row in enumerate(row_lst):
        row_lst[i] = json.loads(row)
        row_lst[i].pop('key')
    return row_lst

def _find_and_read_row(log_str : str, key : str, row_must_exist=True):
    """
    Finds first matching row in a log file string and converts it into a dict.
    """
    regex = r'.*"key": "{}".*'.format(key)
    row = re.search(regex, log_str)
    if row is None:
        if(row_must_exist):
            raise Exception('Failed to match regex: '.format(regex))
        else:
            return None
    row = json.loads(row.group(0))
    row.pop('key')
    return row

## Metrics

def get_exps_metric(log_str : str):
    """
    Given log file in form of loaded in-memory string, calculate
    queries/second
    """
    run_start_row = _find_and_read_row(log_str, constants.RUN_START)
    run_stop_row = _find_and_read_row(log_str, constants.RUN_STOP)

    # calculate runtime
    run_start_time = float(run_start_row['time_ms'])
    run_stop_time = float(run_stop_row['time_ms'])
    seconds_runtime = (run_stop_time - run_start_time) / 1000

    # get num batches and batch size based on available log info
    # batch info in run_stop row to be deprecated
    batch_size_row = _find_and_read_row(log_str, constants.BATCH_SIZE, row_must_exist=False)
    num_batches_row = _find_and_read_row(log_str, constants.NUM_BATCHES, row_must_exist=False)
    if(num_batches_row is not None and batch_size_row is not None):
        num_batches, batch_size = num_batches_row['num_batches'], batch_size_row['batch_size']
    else:
        num_batches, batch_size = run_stop_row['num_batches'], run_stop_row['batch_size']

    # calculate throughput, which is score
    if(seconds_runtime == 0):
        throughput = 'error: runtime is zero'
    else:
        throughput = num_batches * batch_size / seconds_runtime 
    average_batch_time = seconds_runtime / num_batches

    metrics_dict = {'score': throughput, 'units': "ex/s"}
    return metrics_dict

def get_tfps_metric(log_str):
    """
    Given log file in form of loaded in-memory string, calculate
    teraflops/second 
    """
    run_stop_row = _find_and_read_row(log_str, constants.RUN_STOP)
    tfps = run_stop_row['extra_metadata']['TF/s']
    metrics_dict = {'score': tfps, 'units': "TF/s"}
    return metrics_dict

def get_gbps_metric(log_str):
    """
    Given log file in form of loaded in-memory string, calculate
    teraflops/second 
    """
    run_stop_row = _find_and_read_row(log_str, constants.RUN_STOP)
    gbps = run_stop_row['extra_metadata']['GB/s']
    metrics_dict = {'score': gbps, 'units': "GB/s"}
    return metrics_dict

def _calculate_metrics(log_str : str, score_metric : str):
    """
    Calculates metrics. Routes to different metrics functions based on the score_metric type. 
    Allowed score metrics live in loggerconstants.py
    """
    
    # route to correct score_metric, which gets score and units
    if(score_metric == constants.EXPS):
        metrics_dict = get_exps_metric(log_str)
    elif(score_metric == constants.TFPS):
        metrics_dict = get_tfps_metric(log_str)
    elif(score_metric == constants.GBPS):
        metrics_dict = get_gbps_metric(log_str)
    else:
        raise Exception("Score metric not available - should never get here")
    return metrics_dict

## Handle batches and batch latency 

def _calculate_batch_latency(log_str : str, percentile : float):
    """
    Calculates batch latency at a given percentile in range [0, 1]. 
    """
    batch_start_lst = _find_and_read_row_multiple(log_str, constants.BATCH_START)
    batch_stop_lst = _find_and_read_row_multiple(log_str, constants.BATCH_STOP)
    if(len(batch_start_lst) != len(batch_stop_lst)):
        raise Exception('Number of batch starts does not match number of batch stops')
    nbatches = len(batch_start_lst)
    if(nbatches == 0):
        return None

    batch_times = []
    for i in range(nbatches):
        # calculate runtime
        batch_start_time = float(batch_start_lst[i]['time_ms'])
        batch_stop_time = float(batch_stop_lst[i]['time_ms'])
        batch_runtime = (batch_stop_time - batch_start_time) / 1000 # seconds
        batch_times.append(batch_runtime)
    
    batch_times.sort()
    # default to slower latency if percentile doesn't exactly match a batch time
    batch_idx = math.ceil(percentile * nbatches) - 1
    batch_time_at_percentile = batch_times[batch_idx]

    return batch_time_at_percentile

## Read and process log files 

def _create_summary_row(file_path : str):
    """
    Takes a single file path.
    Return JSON row.
    """
    with open(file_path, 'r') as f:
        log_file_str = f.read()
    header = _find_and_read_row(log_file_str, constants.HEADER) 
    metrics = _calculate_metrics(log_file_str, header['score_metric'])
    row = header
    row['metrics'] = metrics
    
    # TODO: allow encoding of extra metadata and include the p95 in the key
    batch_latency = _calculate_batch_latency(log_file_str, 0.95)
    row['batch_latency'] = batch_latency 

    return row

def _rows_to_file(rows: list, folder_path: str, summary_view=constants.INTERMEDIATE_VIEW):
    """
    Save list of summary rows into a human-readable table in a file.
    rows: list[dict]
    """
    file_path = folder_path + '/summary.txt'
    if(len(rows) == 0):
        print('Nothing to summarize, no changes to summary file.')
        return
    open(file_path, 'w') # create or overwrite file

    if(summary_view == constants.INTERMEDIATE_VIEW):
        top_level_keys = [
            "benchmark",
            "implementation",
            "mode",
            "config",
            "score",
            "units",
            "batch_latency"]
        _lst_to_file(top_level_keys, file_path)
        for row in rows:
            flattened_row = _flatten_dict(row)
            top_val_lst = [flattened_row[k] for k in top_level_keys]
            _lst_to_file(top_val_lst, file_path)
    elif(summary_view == constants.RAW_VIEW):
        for row in rows:
            _dump_json(row, file_path)
    else:
        print('Summary view of wrong type - should never get here.')

def summarize_results(benchmark_folder) -> list:
    """
    Summarizes a set of results.
    returns: list[dict]
    """
    rows = []
    pattern = '{folder}/*.log'.format(folder=benchmark_folder) # TODO allow other kinds of files
    result_files = glob.glob(pattern, recursive=True)
    print('Summarizing files: {}'.format(result_files))
    for file_path in result_files:
        row = _create_summary_row(file_path)
        rows.append(row)
    return rows

## Parse and main

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a folder of logged benchmark result files."
    )
    parser.add_argument('-f', '--benchmark-folder', type=str, default='.')
    parser.add_argument('-v', '--summary-view', type=str, default=constants.INTERMEDIATE_VIEW)
    return parser

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    rows = summarize_results(args.benchmark_folder)
    _rows_to_file(rows, args.benchmark_folder, summary_view=args.summary_view)
