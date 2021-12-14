import torch
import sys
import time
import torch.nn.functional as F
import xlmr_data, xlmr_parser

# logging
import pathlib
from os import fspath
p = pathlib.Path(__file__).parent.resolve() / "../../../bmlogging"
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
    fairseq_xlmr_large = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')

    # load model weights file locally
    # f = '/path/xlmr.large'
    # fairseq_xlmr_large = XLMRModel.from_pretrained(f, checkpoint_file='model.pt')

    # TODO use torchscript? jit/script this model?
    return fairseq_xlmr_large

def inference(xlmr, x_l, device=None, logger=None):
    """
    xlmr: xlmr model to infer on
    x_l: data 
    device->torch.device: optional device (generally a gpu). If None, default to cpu.
    logger->BMLogger: optional logger. If no logger, does not log.

    Performs inference loop, with optional logging.
    """
    if logger is None:
        logger = get_bmlogger() #No op logger

    for i, x in enumerate(x_l):
        logger.batch_start()
        if device:
            x = x.to(device) 
        # xlmr.model.encoder.sentence_encoder(x)['encoder_out'][-1] # equivalent
        y_pred = xlmr.extract_features(x) 
        del y_pred 
        logger.batch_stop(time_ms=time_ms(device is not None))

def train(xlmr, x_l, y_true_l, device=None, logger=None):
    """
    xlmr: xlmr model to train
    x_l: input data
    y_true_l: true labels
    device->torch.device: optional device (generally a gpu). If None, default to cpu.
    logger->BMLogger: optional logger. If no logger, does not log.

    Performs train loop, with optional logging.
    """
    if logger is None:
        logger = get_bmlogger() #No op logger

    #training loop
    learning_rate = 0.01
    optimizer = torch.optim.SGD(xlmr.parameters(), lr=learning_rate)
    for i, (x, y_true) in enumerate(zip(x_l, y_true_l)):   
        logger.batch_start()
        if device:
            x = x.to(device)
            y_true = y_true.to(device)
        y_pred = xlmr.extract_features(x)
        loss = F.cross_entropy(y_pred, y_true) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        del y_pred
        del loss
        logger.batch_stop(time_ms=time_ms(device is not None))

def generate_dataset(num_batches, batch_size, vocab_size, inference_only, seqlen_dist=None, uniform_seqlen=None, seq_len_dist_max=256):
    """
    Generates a dataset depending on boolean flags
    inference_only: bool. Whether to return non-empty Y in addition to X. 
    """
    def generate_single_sample():
        get_y_true_arg = not inference_only
        if(seqlen_dist is not None): # distribution takes priority if it exists
            seq_length_arg = xlmr_data.sample_sequence_length(seq_len_dist=seqlen_dist, seq_len_max=seq_len_dist_max)
        elif(uniform_seqlen is not None):
            seq_length_arg = uniform_seqlen
        else:
            raise Exception("Cannot have empty sequence length distribution and uniform sequence length")
        # TODO: Use half kwarg to generate half input data when appropriate
        x, y = xlmr_data.generate_ml_sample(batchsize=batch_size, seq_length=seq_length_arg, vocab_size=vocab_size, get_y_true=get_y_true_arg)
        return x, y

    X_data = []
    Y_data = []
    for _ in range(num_batches):
        x_sample, y_sample = generate_single_sample()
        X_data.append(x_sample)
        if(not inference_only):
            Y_data.append(y_sample)

    return X_data, Y_data

def run():
    parser = xlmr_parser.init_argparse()
    args = parser.parse_args()
    if args.seqlen_dist is not None:
        args.seqlen_dist = xlmr_parser.dict_deserialize(args.seqlen_dist)

    # check for device
    device=None
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

    # use gpu
    if device:
        xlmr = xlmr.to(device)
    if device and args.half_model:
        xlmr.half()
    
    # Generate data! y is empty if inference_only. 
    x_l_warmup, y_true_l_warmup = generate_dataset(args.warmup_batches, args.batch_size, 
        args.vocab_size, args.inference_only, uniform_seqlen=args.sequence_length, 
        seqlen_dist=args.seqlen_dist, seq_len_dist_max=args.seqlen_dist_max)
    x_l, y_true_l = generate_dataset(args.num_batches, args.batch_size, 
        args.vocab_size, args.inference_only, uniform_seqlen=args.sequence_length, 
        seqlen_dist=args.seqlen_dist, seq_len_dist_max=args.seqlen_dist_max)

    # warmup
    if args.inference_only:
        inference(xlmr, x_l_warmup, device=device)
    else:
        train(xlmr, x_l_warmup, y_true_l_warmup, device=device)

    # benchmark!
    bmlogger.run_start(time_ms=time_ms(args.use_gpu))
    
    if args.inference_only:
        inference(xlmr, x_l, device=device, logger=bmlogger)
    else:
        train(xlmr, x_l, y_true_l, device=device, logger=bmlogger)

    bmlogger.run_stop(0, 0, time_ms=time_ms(args.use_gpu))
    bmlogger.record_batch_info(num_batches=len(x_l), batch_size=len(x_l[0]))

if __name__ == "__main__":
    run()
