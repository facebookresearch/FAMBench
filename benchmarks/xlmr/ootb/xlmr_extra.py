"""
Hold functions for xlmr todos
"""
from torchtext.datasets import PennTreebank

def get_inference_data():
    test_dp = PennTreebank(split='test')
    # TODO prepare this data properly 

    return test_dp

def evaluate(test_dp, model):
    """
    evaluation loop for xlmr
    """
    model.eval() 
    total_correct, total_count = 0.0, 0.0
    with torch.no_grad():
        for batch in test_dp:
            print(batch)
            model_input = batch["pad_token_ids"] # TODO .to(device). same for next line
            target = torch.tensor(batch["labels"])
            logits = model(model_input)
            correct = (logits.argmax(1) == target).sum()
            total_correct+=float(correct)
            total_count+=float(target.size(0))
    return total_correct/total_count

def train(self, niter=1):
    # TODO need the right loss, correct optimizer/learning rate, etc
    pass