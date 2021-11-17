import torch
import argparse
import sys
import time
import torch.nn.functional as F

# logging
import pathlib
from os import fspath
p = pathlib.Path(__file__).parent.resolve() / "../../../bmlogger"
sys.path.append(fspath(p))
from bmlogger import get_bmlogger

# from fairseq.models.roberta import XLMRModel

def time_ms(use_gpu):
    """
    Return time. If gpu is available, synchronize.
    """
    if use_gpu:
        torch.cuda.synchronize()
    return time.time_ns() * 1e-6

def get_model():
    # download from Internet
    fairseq_xlmr_large = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')

    # load model weights file locally
    # f = '/path/xlmr.large'
    # fairseq_xlmr_large = XLMRModel.from_pretrained(f, checkpoint_file='model.pt')

    # TODO use torchscript? jit/script this model?
    return fairseq_xlmr_large.model

def generate_ml_sample(batchsize=64, seq_length=64, vocab_size=250000, get_y_true=True):
    shape = (batchsize, seq_length)
    x = torch.rand(shape) * vocab_size
    x = x.int()
    if get_y_true:
        y_true = torch.rand((batchsize, seq_length, 250002))
        return [x, y_true]
    else:
        return x

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark XLM-R model"
    )
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--famconfig", type=str, default="tiny")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=250000)
    parser.add_argument("--half-model", action="store_true", default=False)
    parser.add_argument("--warmup-batches", type=int, default=3)
    return parser

def run():
    parser = init_argparse()
    args = parser.parse_args()

    # check for device
    if(args.use_gpu):
        assert torch.cuda.is_available(), "No cuda device is available."
        device = torch.device("cuda", 0)

    # prep logger
    bmlogger = get_bmlogger() # default to Nop logger
    if args.logdir is not None:
        mode = "train"
        if(args.inference_only):
            mode = "eval"

        logpath = "{}/XLMR_OOTB_{}_{}.log".format(args.logdir, mode, args.famconfig)
        bmlogger = get_bmlogger(logpath)
        bmlogger.header("XLMR", "OOTB", mode, args.famconfig)

    # prep model and data
    xlmr = get_model()
    if args.inference_only:
        xlmr.eval()
    if args.half_model:
        xlmr.half()
    
    # use gpu
    if args.use_gpu:
        xlmr = xlmr.to(device)

    if args.inference_only:
        x_l = [generate_ml_sample(batchsize=args.batch_size, seq_length=args.sequence_length, vocab_size=args.vocab_size, get_y_true=False) for _ in range(args.num_batches)]
    else:
        x_l, y_true_l = zip(*[generate_ml_sample(batchsize=args.batch_size, seq_length=args.sequence_length, vocab_size=args.vocab_size) for _ in range(args.num_batches)])

    # benchmark!
    bmlogger.run_start(time_ms=time_ms(args.use_gpu))
    
    if args.inference_only:
        for x in x_l:
            bmlogger.batch_start()
            if args.use_gpu:
                x = x.to(device)
            y_pred = xlmr(x)
            bmlogger.batch_stop(time_ms=time_ms(args.use_gpu))
    else:
        #training loop
        learning_rate = 0.01
        optimizer = torch.optim.SGD(xlmr.parameters(), lr=learning_rate)
        for x, y_true in zip(x_l, y_true_l):    
            bmlogger.batch_start()
            if args.use_gpu:
                x = x.to(device)
                y_true = y_true.to(device)
            y_pred = xlmr(x)
            y_true = y_true.long()
            loss = F.cross_entropy(y_pred[0], y_true[:,0,:]) # TODO: fix y_true data input hack
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
            bmlogger.batch_stop(time_ms=time_ms(args.use_gpu))

    bmlogger.run_stop(0, 0, time_ms=time_ms(args.use_gpu))
    bmlogger.record_batch_info(num_batches=len(x_l), batch_size=len(x_l[0]))

if __name__ == "__main__":
    run()
