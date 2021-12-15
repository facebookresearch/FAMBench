
import numpy as np

"""
Util functions for XLM-R benchmark
"""

def is_monotonic_increasing(x):
    dx = np.diff(x)
    return np.all(dx >= 0)

def query_from_percentile_distribution(query_percentile, sorted_percentiles_lst, sorted_values_lst):
    """
    Query from a percentile distribution.

    Example: 
    query_percentile = 0.6
    sorted_percentiles_lst = [0, 0.4, 0.8, 1.0]
    sorted_values_lst = [12, 15, 72, 105]
    The query percentile lies right in the middle between p40 (0.4) and p80 (0.8). 
    Thus we linearly interpolate between the two, and get p60 -> (15 + 72) / 2 = 43.5. 
    """

    assert len(sorted_percentiles_lst) == len(sorted_values_lst)
    assert 0 <= query_percentile <= 1
    assert is_monotonic_increasing(sorted_values_lst) 
    assert is_monotonic_increasing(sorted_percentiles_lst) 

    ret_val = None
    for i, p in enumerate(sorted_percentiles_lst):
        if(query_percentile == p):
            ret_val = sorted_values_lst[i] 
            break
        if(query_percentile < p):
            # linearly interpolate between i and i-1th percentile to get right value
            low_p, low_val = sorted_percentiles_lst[i-1], sorted_values_lst[i]
            high_p, high_val = p, sorted_values_lst[i]
            percentile_in_bucket = (query_percentile - low_p) / (high_p - low_p)
            diff_from_low = percentile_in_bucket * (high_val - low_val)
            ret_val = diff_from_low + low_val 
            break
    
    if(ret_val is None):
        # Should never reach here. Asserts should keep input correct so above logic doesn't fail. 
        raise Exception('ret_val should not be None. Should never reach here.')
    
    return ret_val
