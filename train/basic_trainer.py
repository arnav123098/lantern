from utils import dotdict
import torch
import time
from metrics.metrics import Metrics

'''
This trainer takes a model, tokenizer, dataloader, lr_scheduler and optimizer and does the following:
- create train and val datasets
- configure optimizer
- calculate gradient accumulation steps to simulate a larger batch_size
- zero the grads, get loss, perform backward and update params
- calculate val_loss if is_val is True
- print metrics for each step

*This one is kinda tightly coupled - dataloader, training and validation etc. are handled by the trainer.
Later ones will be a lot more modular and framework style.

update: added metrics
'''
class BasicTrainer: # monolithic
    def __init__(self, config): # TODO: add assertions for config
        config = dotdict(config)
        self.config = config
        self.model = config.model

        self.is_val = config.get('is_val', False)
        self.val_interval = config.get('val_interval', 1)
        self.val_batch_size = config.get('val_batch_size', config.batch_size)

        # auto-detect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

        self.model.to(device)

        torch.manual_seed(1337)
        if device == "cuda":
            torch.cuda.manual_seed(1337)

        # gradient accumulation to simulate large batch size
        self.grad_accum_steps = config.batch_size // (config.B*config.T) if config.batch_size else 1
        print(f'gradient accumulation steps: {self.grad_accum_steps}')

        torch.set_float32_matmul_precision('high') # lower precision to speed up training

        self.device = device
        self.optimizer = config.model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=config.max_lr, device=device)
        self.dataloader = config.dataloader(config)

        self.model = torch.compile(self.model)
        self.get_lr = config.lr_scheduler

        self.step = 0 # step state

        self.metrics = Metrics()

    def train(self):
        for step in range(self.config.max_steps):
            t0 = time.time()

            self.optimizer.zero_grad()

            loss_accum = 0.0
            val_loss_accum = 0.0

            # simulating batch_size
            for _ in range(self.grad_accum_steps):
                train_X, train_Y = self.dataloader.next_train_batch()
                train_X, train_Y = train_X.to(self.device), train_Y.to(self.device)
                _, loss = self.model(train_X, targets=train_Y)
                loss /= self.grad_accum_steps # divide to get correct loss for simulated batch_size
                loss_accum += loss.detach()
                loss.backward()
                    
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            val_step = self.is_val and step % self.val_interval == 0

            # validation
            self.model.eval()
            if val_step:
                with torch.no_grad():
                    val_X, val_Y = self.dataloader.next_val_batch()
                    val_X, val_Y = val_X.to(self.device), val_Y.to(self.device)
                    _, val_loss = self.model(val_X, targets=val_Y)
                    val_loss /= self.grad_accum_steps
                    val_loss_accum += val_loss.detach()
            self.model.train()

            # update params
            lr = self.get_lr(self.config, step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()

            torch.cuda.synchronize() # wait for the GPU to finish work

            self.step = step

            # metrics
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = self.config.B * self.config.block_size * self.grad_accum_steps
            if val_step:
              val_tokens_processed = self.config.val_batch_size * self.config.block_size * self.grad_accum_steps
              tokens_processed += val_tokens_processed
            tokens_per_sec = tokens_processed / dt

            if step == 0:
                variables = {
                    'loss': lambda: loss_accum,
                    'lr': lambda: lr,
                    'norm': lambda: norm,
                    'val_loss': lambda: val_loss_accum,
                    'tokens_processed': lambda: tokens_processed,
                    'tokens_per_sec': lambda: tokens_per_sec
                }
                self.metrics.track({
                    k: v
                    for k, v in variables.items()
                    if k in self.config.metrics
                })

                if self.config.extra_metrics is not None:
                    self.metrics.track(self.config.extra_metrics)

            self.metrics.record()

            print(f"step {step:4d} || loss: {loss_accum:.6f} | {f'val_loss: {val_loss_accum:.6f} |' if self.is_val else ''}lr {lr:.8f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

'''
Example usage:
model = GPT2(GPTConfig)
tokenizer = tiktoken.get_encoding('gpt2')

config = {
    'model': model,
    'dataloader': TextLoaderLite,
    'tokenizer': tokenizer,
    'lr_scheduler': cosine_decay,
    'max_lr': 3e-4,
    'min_lr': 3e-5,
    'warmup_steps': 2000,
    'weight_decay': 0.0,
    'max_steps': 500,
    'batch_size': 2**14,
    'B': 4,
    'T': 512,
    'filepath': 'input.txt',
    'val_split': 0.3,
    'is_val': True,
    'val_interval': 10,
    'val_batch_size': 16,
    'metrics': ['loss', 'val_loss'],
    'extra_metrics': {
        'wte_norm': lambda: model.transformer.wte.weight.norm()
    }
}

tr = BasicTrainer(config)
tr.train()

Example output:
using device: cuda
gradient accumulation steps: 8
num decayed parameter tensors: 50, with 124,318,464 parameters
num non-decayed parameter tensors: 98, with 121,344 parameters
step    0 || loss: 10.788390 | val_loss: 10.795937 |lr 0.00003000 | norm: 10.3416 | dt: 11526.95ms | tok/sec: 2842.73
step    1 || loss: 10.183259 | val_loss: 10.238352 |lr 0.00003000 | norm: 6.0999 | dt: 4595.12ms | tok/sec: 7131.05
...
'''
