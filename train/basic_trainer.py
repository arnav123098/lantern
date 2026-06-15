from lantern.utils import dotdict
import torch
import time
from lantern.metrics import Metrics

'''
BasicTrainer which is not so basic:
- setup dataloaders for train and val
- set device
- set precision
- other efficiency tricks
- configure optimizers
- calculate gradient accumulation steps to simulate a larger batch size (using accum_size)
- zero the grads, get loss, perform backward and update params
- calculate val_loss if is_val is True
- print metrics for each step
'''
class BasicTrainer:
    def __init__(self, config: dict):
        config = dotdict(config)
        self.config = config

        self.model = config.model

        self.seed = config.seed if config.seed is not None else 1111

        # setup loaders
        self.train_loader = config.train_loader
        self.val_loader = config.val_loader # else None

        # setup val interval
        self.is_val = config.get('is_val', False)
        self.val_interval = config.get('val_interval', 1)

        # auto-detect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

        self.model.to(device)

        torch.manual_seed(self.seed)
        if device == "cuda":
            torch.cuda.manual_seed(self.seed)

        # gradient accumulation to simulate large batch size
        self.accum_size = config.accum_size if config.accum_size is not None else self.train_loader.B

        self.grad_accum_steps = self.accum_size // (self.train_loader.B*self.train_loader.T)
        print(f'gradient accumulation steps: {self.grad_accum_steps}')

        torch.set_float32_matmul_precision(config.float32_matmul_precision or 'high') # lower precision to speed up training

        self.device = device

        self.optimizer = config.model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=config.min_lr, device=device)

        self.model = torch.compile(self.model)
        self.lr_scheduler = config.lr_scheduler

        self.step = 0 # step state

        self.metrics = Metrics()

    def train(self):
        for step in range(self.step, self.config.max_steps):
            t0 = time.time()

            self.optimizer.zero_grad()

            loss_accum = 0.0
            val_loss = 0.0
            
            # simulating accum_size
            for _ in range(self.grad_accum_steps):
                train_X, train_Y = self.train_loader.next_batch()
                train_X, train_Y = train_X.to(self.device), train_Y.to(self.device)
                _, loss = self.model(train_X, targets=train_Y)
                loss /= self.grad_accum_steps # divide to get correct loss for simulated larger batch size
                loss_accum += loss.detach()
                loss.backward()
            
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            val_step = self.is_val and step % self.val_interval == 0

            # validation
            if val_step:
                self.model.eval()
                with torch.no_grad():
                    val_X, val_Y = self.val_loader.next_batch()
                    val_X, val_Y = val_X.to(self.device), val_Y.to(self.device)
                    _, val_loss = self.model(val_X, targets=val_Y)
                    val_loss = val_loss.detach()
                self.model.train()

            # update params
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_lr(step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            self.optimizer.step()

            if self.device == "cuda":
                torch.cuda.synchronize() # wait for the GPU to finish work

            self.step = step # save step state

            # metrics
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps
            if val_step:
              val_tokens_processed = self.val_loader.B * self.val_loader.T
              tokens_processed += val_tokens_processed
            tokens_per_sec = tokens_processed / dt

            if step == 0: # register variables to track
                variables = {
                    'loss': lambda: loss_accum,
                    'lr': lambda: lr,
                    'norm': lambda: norm,
                    'val_loss': lambda: val_loss,
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

            self.metrics.record() # record metrics

            print(f"step {step:4d} || loss: {loss_accum:.6f} | {f'val_loss: {val_loss:.6f} |' if val_step else ''}lr {lr:.8f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

'''
Example usage:
config = {
    'model': model,
    'train_loader': TextLoader(...),
    'val_loader': TextLoader(...),
    'lr_scheduler': CosineDecay(...),
    'weight_decay': 0.0,
    'max_steps': 500,
    'accum_size': 2**14,
    'is_val': True,
    'val_interval': 10,
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
