from benchmark import Benchmark
from data.hellaswag_loader import HellaswagLoader
from utils import dotdict
import torch.nn.functional as F
import torch

'''
This hellaswag evaluator uses our HellaswagLoader to load batches of questions and evaluate the model.
What this does:
- get logits
- calculate loss (reduction='none')
- apply completion mask
- get the lowest loss and calculate accuracy

TODO: testing
TODO (for later): attn_mask 
'''
class Hellaswag(Benchmark):
    def __init__(self, config: dotdict):
        super().__init__(config)
        self.loader = HellaswagLoader(config)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()

        acc = 0
        while True:
            batch = self.loader.next_batch()
            if batch is None: break

            X, Y = batch['batch']
            mask = batch['mask']

            B, C, T = X.shape
            X = X.view(B * C, T)

            # calculating loss
            logits, _ = self.model(X)
            logits = logits[:, :-1, :]

            targets = X[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                reduction='none'
            )
            loss = loss.reshape(B, C, T - 1)

            mask = mask[:, :, 1:] # (B, C, T - 1)
            loss *= mask
            avg_losses = loss.sum(-1) / mask.sum(-1).clamp(min=1) # (B, C)
            preds = avg_losses.argmin(dim=-1) # (B)

            acc += (preds == Y).sum().item()

            accuracy = acc / len(self.loader)
            return accuracy
