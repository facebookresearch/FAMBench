

import torch


# FB5 Logger
import pathlib
from os import fspath
p = pathlib.Path(__file__).parent.resolve() / "../../../fb5logging"
sys.path.append(fspath(p))
from fb5logger import FB5Logger

def import_dataset():
    pass
    # TODO import the right dataset

def eval(self, niter=1):
    trainer = self.trainer
    for _ in range(niter):
        # 1. forward the next_sentence_prediction and masked_lm model
        next_sent_output, mask_lm_output = trainer.model.forward(*self.example_inputs)

        # 2-1. NLL(negative log likelihood) loss of is_next classification result
        # 2-2. NLLLoss of predicting masked token word
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        next_loss = trainer.criterion(next_sent_output, self.is_next)
        mask_loss = trainer.criterion(mask_lm_output.transpose(1, 2), self.bert_label)
        loss = next_loss + mask_loss

def train(self, niter=1):
    # TODO need the right loss, correct optimizer/learning rate, etc

    trainer = self.trainer
    for _ in range(niter):
        # 1. forward the next_sentence_prediction and masked_lm model
        next_sent_output, mask_lm_output = trainer.model.forward(*self.example_inputs)

        # 2-1. NLL(negative log likelihood) loss of is_next classification result
        # 2-2. NLLLoss of predicting masked token word
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        next_loss = trainer.criterion(next_sent_output, self.is_next)
        mask_loss = trainer.criterion(mask_lm_output.transpose(1, 2), self.bert_label)
        loss = next_loss + mask_loss

        # 3. backward and optimization only in train
        trainer.optim_schedule.zero_grad()
        loss.backward()
        trainer.optim_schedule.step_and_update_lr()


def run():
    # TODO parse args for --inference only, configs, etc
    
    dataset = import_dataset() 

    # TODO setup the XLM-R model. Pick the right one - large, xl, xxl? 
    xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.xl')

    if args.fb5logger is not None:
        fb5logger = FB5Logger(args.fb5logger)
        if args.inference_only:
            fb5logger.header("XLMR", "OOTB", "eval", args.fb5config)
        else:
            fb5logger.header("XLMR", "OOTB", "train", args.fb5config)
            
    if args.fb5logger is not None:
            fb5logger.run_start()

    if(): # not inference_only -> train it 
        xlmr.train()
    else:
        xlmr.eval()  # disable dropout (or leave in train mode to finetune)

    if args.fb5logger is not None:
        fb5logger.run_stop(nbatches, args.mini_batch_size)