import argparse
import json

"""
Parse inputs for the XLMR benchmark
"""

def parse_seqlen_dist(seqlen_dist_str):
    seqlen_dist_json = json.loads(seqlen_dist_str)

    # assert that it is in the right format (percentiles and integers)

    return seqlen_dist_json


def init_argparse() -> argparse.ArgumentParser:
    """
    Returns a parser that can parse the given inputs by calling parser.parse_args()
    
    Some types are functions - this is because argparse under the hood just calls the type on the input string. 
    Eg if type=int, then if you get a str="123", argparse calls int("123")->123. 
    """

    parser = argparse.ArgumentParser(
        description="Benchmark XLM-R model"
    )
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--famconfig", type=str, default="tiny")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--num-batches", type=int, default=10) # num batches to benchmark 
    parser.add_argument("--warmup-batches", type=int, default=0) # num batches to warmup
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=64) 
    parser.add_argument("--vocab-size", type=int, default=250000)
    parser.add_argument("--half-model", action="store_true", default=False)
    parser.add_argument('--seqlen-dist', type=str, default=None) # sequence length distribution. Type is string in JSON format. 
    parser.add_argument('--seqlen-dist-max', type=int, default=256) # maximum allowed sequence length
    
    return parser

