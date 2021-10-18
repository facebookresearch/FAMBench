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
    
def evaluate_simple(model, input_data, famlogger=None):
    """
    Run data through the model
    """
    for batch in input_data:
        famlogger.batch_start()
        output = model(batch)
        famlogger.batch_stop()

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark XLM-R model"
    )
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--fb5config", type=str, default="tiny")
    return parser

def run():
    parser = init_argparse()
    args = parser.parse_args()

    # prep logger
    if args.logfile is not None:
        famlogger = FB5Logger(args.logfile)
        if(args.inference_only): 
            famlogger.header("XLMR", "OOTB", "eval", args.fb5config)
        else:
            famlogger.header("XLMR", "OOTB", "train", args.fb5config)
            
    # prep model and data
    xlmr = None
    data = None
    if(args.inference_only): 
        data = generate_inference_data(nbatches=10)
        xlmr = get_inference_model()
    else:
        pass # TODO train side

    # TODO: run timer sync for gpu torch.cuda.synchronize(). run .cuda() on model and data here. 
    # Use .half() on the model to use fp16 instead of 32. 2x size!

    # benchmark! 
    if args.logfile is not None:
        famlogger.run_start()

    evaluate_simple(xlmr, data, famlogger=famlogger) 

    if args.logfile is not None:
        famlogger.run_stop(1, 1)    
        famlogger.record_batch_info(num_batches=data.shape[0], batch_size=data.shape[1])

if __name__ == "__main__":
    run()
