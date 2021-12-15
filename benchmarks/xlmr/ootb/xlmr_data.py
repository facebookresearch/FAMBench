import torch
import numpy as np
import xlmr_utils
"""
Data generation for XLM-R benchmark
"""

def sample_sequence_length(query_percentile = None, seq_len_dist=None, seq_len_max=float('inf')):
    # TODO This kind of code probably exists somewhere? Fun to code but find a replacement.
    # TODO Benchmark may OOM stochastically - reduce it
    """
    Given a percentile, gives the corresponding sequence length based on seq len distribution.
    If percentile is none, percentile is randomly generated.

    Expects 
    query_percentile : float in range [0, 1]
    seq_len_dist : dict with float keys in range [0, 1] and int vals in range [1, positive inf]
    seq_len_max : int in range [1, positive inf]
    """
    def enforce_bounds(val, min_val, max_val):
        """
        Enforces max and min val on the value
        """
        val = min(val, max_val)
        val = max(val, min_val)
        return val

    if(query_percentile is None):
        query_percentile = np.random.rand()

    if(seq_len_dist is None): # default is to always return 64 seq len
        seq_len_dist = {
            1 : 64,
            0 : 64,
        }

    sorted_percentiles = sorted(seq_len_dist)
    sorted_seqlen = [seq_len_dist[sorted_percentiles[i]] for i in range(len(sorted_percentiles))]

    seq_length_sample = xlmr_utils.query_from_percentile_distribution(query_percentile, sorted_percentiles, sorted_seqlen)
    seq_length_sample = int(seq_length_sample)
    seq_length_sample = enforce_bounds(seq_length_sample, 1, seq_len_max)
    return seq_length_sample

def generate_inputs(batchsize, seq_length, vocab_size, is_half=False):
    shape = (batchsize, seq_length)
    x = torch.rand(shape) * vocab_size
    x = x.int()
    if is_half:
        x = x.half()
    return x

def generate_outputs(batchsize, seq_length, output_embed_size):
    shape = (batchsize, seq_length, output_embed_size)
    y = torch.rand(shape)
    return y 

def generate_ml_sample(batchsize=64, seq_length=64, vocab_size=250000, get_y_true=True, is_half=False):
    """
    Generates data for XLMR benchmark
    """
    x = generate_inputs(batchsize, seq_length, vocab_size, is_half=is_half)
    y = torch.tensor([])
    if(get_y_true):
        output_embed_size = 1024
        y = generate_outputs(batchsize, seq_length, output_embed_size)
        
    return x, y
