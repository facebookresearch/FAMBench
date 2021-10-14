import torch
import argparse
import sys

# FB5 Logger
import pathlib
from os import fspath
p = pathlib.Path(__file__).parent.resolve() / "../../../fb5logging"
sys.path.append(fspath(p))
from fb5logger import FB5Logger

def get_inference_model():
    fairseq_xlmr_large = torch.hub.load('pytorch/fairseq:main', 'xlmr.large') 
    fairseq_xlmr_large.eval()
    # TODO use torchscript? jit/script this model?
    return fairseq_xlmr_large.model 

def generate_inference_data(nbatches=10, batchsize=32, seq_length=64, vocab_size=1000):
    shape = (nbatches, batchsize, seq_length)
    data = torch.rand(shape) * vocab_size
    data = data.int()
    return data
    
def evaluate_simple(model, input_data):
    """
    Run data through the model
    """
    for batch in input_data:
        output = model(batch)

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark XLM-R model"
    )
    parser.add_argument("--fb5logger", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--fb5config", type=str, default="tiny")
    return parser

def run():
    parser = init_argparse()
    args = parser.parse_args()

    # prep logger
    if args.fb5logger is not None:
        fb5logger = FB5Logger(args.fb5logger)
        if(args.inference_only): 
            fb5logger.header("XLMR", "OOTB", "eval", args.fb5config)
        else:
            fb5logger.header("XLMR", "OOTB", "train", args.fb5config)
            
    # prep model and data
    xlmr = None
    data = None
    if(args.inference_only): 
        data = generate_inference_data()
        xlmr = get_inference_model()
    else:
        pass # TODO train side

    # benchmark! 
    if args.fb5logger is not None:
        fb5logger.run_start()

    evaluate_simple(xlmr, data) 

    nbatches, batch_size = data.shape[0], data.shape[1]
    if args.fb5logger is not None:
        fb5logger.run_stop(nbatches, batch_size)    

if __name__ == "__main__":
    run()
