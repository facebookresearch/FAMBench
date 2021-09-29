import torch

from fairseq.models.roberta import XLMRModel

# FB5 Logger
import pathlib
from os import fspath
p = pathlib.Path(__file__).parent.resolve() / "../../../fb5logging"
sys.path.append(fspath(p))
from fb5logger import FB5Logger

import torch.utils.data.datapipes as dp
from torchdata.datapipes.iter import ZipArchiveReader
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
)

#download files to local_path from this URL
URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
local_path = "" # TODO make this the right path 

@_wrap_split_argument(("train", "dev", "test"))
def SST2(root, split):
    loader_dp = dp.iter.FileLoader([local_path])
    extracted_files = ZipArchiveReader(loader_dp)
    filter_extracted_files = extracted_files.filter(lambda x: split in x[0])
    return filter_extracted_files.parse_csv(skip_header=True, delimiter="\t").map(
        lambda x: (x[0], int(x[1]))
    )

def process_dp(input_dp, pre_processor,batch_size):
    # TODO: Convert datapipe to dataframe as proposed in Option 4 during Hack-week: Reference N1073381, N1135193
    input_dp = input_dp.map(lambda x: (pre_processor(x[0]),x[1]))
    input_dp = input_dp.batch(batch_size).rows2columnar(["token_ids","labels"])
    input_dp = input_dp.map(lambda x: {"pad_token_ids": pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in x["token_ids"]],\
                            batch_first=True,padding_value=pre_processor.padding_idx),"labels":x["labels"]})
    return input_dp

def setup_inference():
    """
    Returns inference xlmr model and dataset
    """
    
    fairseq_xlmr_large = torch.hub.load('pytorch/fairseq', 'xlmr.large')
    fairseq_xlmr_large.eval()

    test_dp = process_dp(SST2(split='dev'), xlmr_processor, batch_size)

    return fairseq_xlmr_large, test_dp
    # TODO use torchscript?

def evaluate(test_dp, model):
    """
    evaluation loop for xlmr
    """
    model.eval() 
    total_correct, total_count = 0.0, 0.0
    with torch.no_grad():
        for batch in test_dp:
            model_input = batch["pad_token_ids"].to(device)
            target = torch.tensor(batch["labels"]).to(device)
            logits = model(model_input)
            correct = (logits.argmax(1) == target).sum()
            total_correct+=float(correct)
            total_count+=float(target.size(0))
    return total_correct/total_count

def train(self, niter=1):
    # TODO need the right loss, correct optimizer/learning rate, etc
    pass


def run():
    # TODO parse args for --inference only, configs, etc
    
    dataset = import_dataset() 

    if args.fb5logger is not None:
        fb5logger = FB5Logger(args.fb5logger)
        if args.inference_only:
            fb5logger.header("XLMR", "OOTB", "eval", args.fb5config)
        else:
            fb5logger.header("XLMR", "OOTB", "train", args.fb5config)
            
    if args.fb5logger is not None:
        fb5logger.run_start()

    if(not args.inference_only): 
        pass # TODO train side
    else:
        xlmr, test_dp = setup_inference()
        accuracy = evaluate(test_dp, xlmr_classifier_base, xlmr_processor)

    if args.fb5logger is not None:
        fb5logger.run_stop(nbatches, args.mini_batch_size)