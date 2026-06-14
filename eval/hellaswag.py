from lantern.eval.benchmark import Benchmark
from lantern.data.load.hellaswag_loader import HellaswagLoader
from lantern.utils import dotdict
import torch.nn.functional as F
import torch

'''
This hellaswag evaluator uses our HellaswagLoader to load batches of questions and evaluate the model.
What this does:
- get logits
- calculate loss (reduction='none')
- apply completion mask
- get the lowest loss and calculate accuracy

TODO (for later): attn_mask, progress bar
'''
class Hellaswag(Benchmark):
    def __init__(self, config: dotdict):
        super().__init__(config)
        self.loader = HellaswagLoader(config)
        self.device = config.device if config.device is not None else 'cpu'

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()

        acc = 0
        while True:
            batch = self.loader.next_batch()
            if batch is None: break

            X, Y = batch['batch']
            X, Y = X.to(self.device), Y.to(self.device)

            mask = batch['mask']
            mask = mask.to(self.device)

            B, C, T = X.shape
            X = X.view(B * C, T)

            # calculating loss
            logits, _ = self.model(X)
            logits = logits[:, :-1, :]

            targets = X[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, self.model.config.vocab_size),
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

'''
Usage:
config = dotdict({
    'model': model,
    'batch_size': 64,
    'tokenizer': tokenizer,
    'pad_token': pad_token_idx,
    'device': device
})

h = Hellaswag(config)
h.evaluate()
'''
