from lantern.data.load.dataloader import DataLoader
from lantern.metrics import Metrics
from lantern.sched.lr_scheduler import LRScheduler
from lantern.optim.optimizer import Optimizer
from lantern.checkpoint import Checkpoint
from lantern.utils import GPU_FLOPS, estimate_mfu

import torch
import torch.nn as nn

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
import time
from typing import Literal

@dataclass
class TrainerConfig:
    model: nn.Module
    optimizers: Optimizer | tuple[Optimizer, ...] | Callable # can be a function that returns configured_optimizer (i'll make a class for it later)
    train_loader: DataLoader
    max_steps: int

    model_name: str | None = None

    accum_size: int | None = None

    metrics: list | None = None
    extra_metrics: dict | None = None
    peak_flops: float | int | None = None # for mfu

    weight_decay: float = 0.0

    lr_schedulers: LRScheduler | tuple[LRScheduler, ...] | None = None
    lr: float | tuple[float, ...] | None = None

    # precision
    float32_matmul_precision: Literal["highest", "high", "medium"] = "high"
    mixed_precision: Literal["none", "bf16", "fp16"] = "none"

    device: Literal["cpu", "cuda", "mps"] | None = None
    compile: bool = True

    # val setup
    is_val: bool = False
    val_interval: int = 1
    val_loader: DataLoader | None = None

    # to save every n_steps
    n_save_steps: int | None = None
    autosave: bool = True
    save_path: str | None = None

    seed: int = 69

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

        # setup loaders
        self.train_loader = config.train_loader
        self.val_loader = config.val_loader

        self.dataloaders = [self.train_loader] + ([self.val_loader] if self.val_loader is not None else [])

        # setup val interval
        self.is_val = config.is_val
        if self.is_val:
            assert self.val_loader is not None

        self.val_interval = config.val_interval

        self.device = config.device

        if self.device is None:
            # auto-detect device
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"

            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"

            print(f"using device: {self.device} (auto-detected)")
        else:
            print(f"using device: {self.device}")

        # seed
        self.seed = config.seed
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(self.seed)

        # setup model and optimizers
        self.raw_model = config.model

        self.model = config.model
        self.model_name = config.model_name if config.model_name is not None else self.model.__class__.__name__

        self.model.to(self.device)

        assert config.lr_schedulers is not None or config.lr is not None, "In case of no LR scheduler, config.lr is required"

        self.lr_schedulers = tuple() if config.lr_schedulers is None else config.lr_schedulers
        if not isinstance(self.lr_schedulers, tuple): self.lr_schedulers = (self.lr_schedulers,)

        fixed_lr = tuple() if config.lr is None else config.lr
        if not isinstance(fixed_lr, tuple): fixed_lr = (fixed_lr,)

        self.get_lr = [sched.get_lr for sched in self.lr_schedulers]
        self.get_lr += [lambda step, lr=lr: lr for lr in fixed_lr]

        init_lr = [sched.min_lr for sched in self.lr_schedulers]
        init_lr += list(fixed_lr)
        init_lr = init_lr[0] if len(init_lr) == 1 else init_lr

        self.optimizers = config.optimizers

        if isinstance(self.optimizers, Callable):
            self.optimizers = self.optimizers(weight_decay=config.weight_decay, learning_rate=init_lr, device=self.device)

        if not isinstance(self.optimizers, tuple):
            self.optimizers = (self.optimizers,)

        assert len(self.optimizers) == len(self.get_lr) # get_lr has functions that return lr for each optimizer at each step

        # gradient accumulation to simulate large batch size
        self.accum_size = config.accum_size if config.accum_size is not None else self.train_loader.B * self.train_loader.T

        assert self.accum_size % (self.train_loader.B * self.train_loader.T) == 0

        self.grad_accum_steps = self.accum_size // (self.train_loader.B*self.train_loader.T)
        print(f'gradient accumulation steps: {self.grad_accum_steps}')

        # compile model to speed things up
        if config.compile:
            self.model = torch.compile(self.model)

        self.step = 0 # step state
        self.start_step = 0 # 0 or checkpoint step if resuming

        self.metrics = Metrics()

        # precision
        torch.set_float32_matmul_precision(config.float32_matmul_precision)

        if config.mixed_precision == "bf16":
            self.autocast_dtype = torch.bfloat16
            self.scaler = None
            self.precision = "bf16"
        elif config.mixed_precision == "fp16":
            self.autocast_dtype = torch.float16
            self.scaler = torch.amp.GradScaler(self.device)
            self.precision = "fp16"
        else:
            self.autocast_dtype = None
            self.scaler = None
            self.precision = "fp32"

        self.is_autocast = self.autocast_dtype is not None and self.device != "cpu"

        self.peak_flops = config.peak_flops
        if config.peak_flops is None:
            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name()
                if gpu_name in GPU_FLOPS:
                    self.peak_flops = GPU_FLOPS[gpu_name][self.precision]
                    print(f"detected device {gpu_name} | peak flops: {int(self.peak_flops)}")
                else:
                    self.peak_flops = None

        # auto-saving checkpoints
        self.n_save_steps = config.n_save_steps
        self.save_path = config.save_path
        self.autosave = config.autosave

        if config.autosave:
            if self.n_save_steps is None or self.n_save_steps > config.max_steps:
                self.n_save_steps = config.max_steps

            if self.save_path is None:
                self.save_path = str(Path.home() / '.lantern' / 'checkpoints' / self.model_name)

                Path(self.save_path).parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

        print(f"eval: {self.is_val}")

    def state_dict(self):
        return {
            "step": self.step,
            "seed": self.seed,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None
        }

    def load_state_dict(self, state_dict: dict):
        torch.set_rng_state(
            state_dict["rng_state"].cpu()
        )

        if state_dict["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(
                [t.cpu() for t in state_dict["cuda_rng_state"]]
            )

        self.step = state_dict.get('step', 0)
        self.start_step = self.step
        self.seed = state_dict.get('seed', self.seed)

        if self.scaler is not None and state_dict['scaler']:
            self.scaler.load_state_dict(state_dict['scaler'])

    def load_checkpoint(self, path: str):
        Checkpoint.load(
            path=path,
            model=self.model,
            optimizers=self.optimizers,
            dataloaders=self.dataloaders,
            trainer=self,
            device=self.device
        )

        print(f"Checkpoint loaded from path {path}")

    def train(self):
        for step in range(self.start_step, self.config.max_steps):
            t0 = time.time()

            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)

            loss_accum = 0.0
            val_loss = 0.0

            # simulating accum_size
            for _ in range(self.grad_accum_steps):
                train_X, train_Y = self.train_loader.next_batch()
                train_X, train_Y = train_X.to(self.device), train_Y.to(self.device)

                with torch.autocast(
                    device_type=self.device,
                    dtype=self.autocast_dtype,
                    enabled=self.is_autocast
                ):
                    _, loss = self.model(train_X, targets=train_Y)
                    loss /= self.grad_accum_steps # divide to get correct loss for simulated larger batch size

                loss_accum += loss.detach().float() # accumulate losses in float to prevent error stacking up due to lost precision

                if self.scaler is None:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()

            if self.scaler is not None:
                for optimizer in self.optimizers:
                    self.scaler.unscale_(optimizer)

            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # validation
            val_step = self.is_val and step % self.val_interval == 0

            if val_step:
                self.model.eval()
                with torch.no_grad():
                    val_X, val_Y = self.val_loader.next_batch()
                    val_X, val_Y = val_X.to(self.device), val_Y.to(self.device)

                    with torch.autocast(
                        device_type=self.device,
                        dtype=self.autocast_dtype,
                        enabled=self.is_autocast
                    ):
                        _, val_loss = self.model(val_X, targets=val_Y)

                    val_loss = val_loss.detach().float()

                self.model.train()

            # optimizer step
            lr = [lr(step) for lr in self.get_lr]
            for optimizer, optim_lr in zip(self.optimizers, lr):
                for param_group in optimizer.param_groups:
                        param_group['lr'] = optim_lr

                if self.scaler is None:
                    optimizer.step()
                else:
                    self.scaler.step(optimizer)

            # update scaler
            if self.scaler is not None:
                self.scaler.update()

            if self.device == "cuda":
                torch.cuda.synchronize() # wait for the GPU to finish work

            self.step = step + 1 # save step state

            # metrics
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps
            if val_step:
              val_tokens_processed = self.val_loader.B * self.val_loader.T
              tokens_processed += val_tokens_processed
                
            tokens_per_sec = tokens_processed / dt

            mfu_estimate = None
            if self.peak_flops is not None:
                mfu_estimate = estimate_mfu(model=self.model, tokens_per_sec=tokens_per_sec, precision=self.precision, peak_flops=self.peak_flops)
            elif step == 0:
                print("Skipping mfu estimate as peak_flops cannot be calculate. Consider providing it explictly if you want mfu calculation.")
                
            if step == self.start_step: # register variables to track
                variables = {
                    'loss': lambda: loss_accum,
                    'lr': lambda: lr,
                    'norm': lambda: norm,
                    'tokens_processed': lambda: tokens_processed,
                    'tokens_per_sec': lambda: tokens_per_sec,
                    'dt': lambda: dt,
                    'mfu_estimate': lambda: mfu_estimate,
                    'sys': lambda: None # metrics tracks sys internally
                }

                if self.config.metrics is not None:
                    self.metrics.track({
                        k: v
                        for k, v in variables.items()
                        if k in self.config.metrics
                    })

                if self.config.extra_metrics is not None:
                    self.metrics.track(self.config.extra_metrics)

            self.metrics.record() # record metrics

            # auto-save checkpoint
            save_step = self.autosave and (self.step % self.n_save_steps == 0 or self.step == self.config.max_steps)
            print(f"step {step:4d} || loss: {loss_accum:.6f} | {f'val_loss: {val_loss:.6f} |' if val_step else ''}lr: {lr[0] if len(lr) == 1 else lr} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

            if save_step:
                print(f'Saving checkpoint for: {self.step} steps')

                path = f'{self.save_path}__{self.step}_steps.pt'

                Checkpoint.save(
                    path=path,
                    step=self.step,
                    model=self.raw_model,
                    optimizers=self.optimizers,
                    dataloaders=self.dataloaders,
                    trainer=self
                )

                print(f'Checkpoint saved at {path}')
