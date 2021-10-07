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

## Metrics

def get_qps_metric(log_str : str):
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

    num_batches, batch_size = run_stop_row['num_batches'], run_stop_row['batch_size']

    # calculate throughput, which is score
    throughput = num_batches * batch_size / seconds_runtime # TODO if these divisons are by 0 catch exception
    average_batch_time = seconds_runtime / num_batches

    metrics_dict = {'score': throughput, 'units': "ex/sec"}
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

def _find_and_read_row(result : str, key : str):
    """
    Finds a single row in a log file string and converts it into a dict.
    """
    regex = r'.*"key": "{}".*'.format(key)
    row = re.search(regex, result)
    if row is None:
      raise Exception('Failed to match regex: '.format(regex))
    row = json.loads(row.group(0))
    row.pop('key')
    return row

def _calculate_metrics(log_str : str, score_metric : str):
    """
    Calculates results dictionary with the following keys:
      score = examples / sec
      num_batches = number of batches
      batch_size = number of examples per batch
      average_batch_time = seconds / batch 
      extra_metadata = any extra information - may be used for future summary info
    """
    
    # route to correct score_metric, which gets score and units
    if(score_metric == constants.QPS):
        metrics_dict = get_qps_metric(log_str)
    elif(score_metric == constants.TFPS):
        metrics_dict = get_tfps_metric(log_str)
    elif(score_metric == constants.GBPS):
        metrics_dict = get_gbps_metric(log_str)
    else:
        raise Exception("Score metric not available - should never get here")
    return metrics_dict

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
        log_file_str = f.read()
    header = _find_and_read_row(log_file_str, constants.HEADER) 
    metrics = _calculate_metrics(log_file_str, header['score_metric'])

    row = header
    row['metrics'] = metrics
    return row

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

def _rows_to_file(rows: list[dict], folder_path: str, summary_view=constants.INTERMEDIATE_VIEW):
    """
    Save list of summary rows into a human-readable table in a file
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
            "units"]
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

def summarize_results(benchmark_folder) -> list[dict]:
    """
    Summarizes a set of results.
    """
    rows = []
    pattern = '{folder}/*.log'.format(folder=benchmark_folder) # TODO allow other kinds of files
    result_files = glob.glob(pattern, recursive=True)
    print('Summarizing files: {}'.format(result_files))
    for file_path in result_files:
        row = _create_summary_row(file_path)
        rows.append(row)
    return rows

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
