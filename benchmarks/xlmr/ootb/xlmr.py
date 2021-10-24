import torch
import argparse
import sys
import time

# FB5 Logger
import pathlib
from os import fspath
p = pathlib.Path(__file__).parent.resolve() / "../../../fb5logging"
sys.path.append(fspath(p))
from fb5logger import get_fb5logger

def time_ms(use_gpu):
    """
    Return time. If gpu is available, synchronize.
    """
    if use_gpu:
        torch.cuda.synchronize()
    return time.time_ns() * 1e-6

def get_inference_model():
    fairseq_xlmr_large = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
    fairseq_xlmr_large.eval()
    fairseq_xlmr_large.half() # TODO make this a command line arg
    # TODO use torchscript? jit/script this model?
    return fairseq_xlmr_large.model

def generate_inference_data(nbatches=100, batchsize=64, seq_length=64, vocab_size=1000):
    shape = (nbatches, batchsize, seq_length)
    data = torch.rand(shape) * vocab_size
    data = data.int()
    return data

def evaluate_simple(model, input_data, fb5logger=None):
    """
    Run data through the model
    """
    for batch in input_data:
        fb5logger.batch_start()
        output = model(batch)
        fb5logger.batch_stop()

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark XLM-R model"
    )
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--famconfig", type=str, default="tiny")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    return parser

def run():
    parser = init_argparse()
    args = parser.parse_args()

    # check for device
    if(args.use_gpu):
        assert torch.cuda.is_available(), "No cuda device is available."
        device = torch.device("cuda", 0)

    # prep logger
    fb5logger = get_fb5logger(args.logfile)
    if(args.inference_only):
        fb5logger.header("XLMR", "OOTB", "eval", args.famconfig)
    else:
        fb5logger.header("XLMR", "OOTB", "train", args.famconfig)

    # prep model and data
    xlmr = None
    data = None
    if(args.inference_only):
        data = generate_inference_data()
        xlmr = get_inference_model()
    else:
        pass # TODO train side

    # use gpu
    if args.use_gpu:
        data = data.to(device)
        xlmr = xlmr.to(device)

    # benchmark!
    fb5logger.run_start(time_ms=time_ms(args.use_gpu))

    evaluate_simple(xlmr, data, fb5logger=fb5logger)

    fb5logger.run_stop(0, 0, time_ms=time_ms(args.use_gpu))
    fb5logger.record_batch_info(num_batches=data.shape[0], batch_size=data.shape[1])

if __name__ == "__main__":
    run()
