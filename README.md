<pre>
██╗      █████╗ ███╗   ██╗████████╗███████╗██████╗ ███╗   ██╗
██║     ██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔══██╗████╗  ██║
██║     ███████║██╔██╗ ██║   ██║   █████╗  ██████╔╝██╔██╗ ██║
██║     ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██║╚██╗██║
███████╗██║  ██║██║ ╚████║   ██║   ███████╗██║  ██║██║ ╚████║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝
</pre>
### v1: GPT 2/3 era (done)
GPT2, dataloading, training, metrics and plots, checkpointing, loading weights, basic inference and hellaswag eval; long story short, a comprehensive, basic training stack. Lantern has also got many optimizer implementations including SGD (+ momentum), RMSProp, Adam, AdamW and Lion.

### v2: Llama era (done)
Tinyllama implemented and tested (RoPE, GQA and MQA), Muon and MuonW optimizers, ShardsLoader (dataloader for loading large datasets), Trainer (a big upgrade over BasicTrainer - mixed precision, multiple optims and schedulers, more robust, auto-saving etc.)

### v2.5: Inference-focused
KV Cache, Sampling with TopK and Temperature args, Batch inference with padding and padding-mask, refinements and fixes in other areas - Tokenizer class, Model class, concurrent downloads, batch encode tokens, etc.

### v3: Deepseek MoE (upcoming)
...

## TODOS:
- [ ] Cleaner documentation

---
---
### note:
i plan to build this out slowly over a month or two <br>
this is gonna be for my learning as well as to serve as a nice educational resource

![random gif](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzR2bXJjbDY4MHIwNmV5NHF2bWlxd3N6OHl4d2lvcml4bmhrbDhmbSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QMOdMJIox7NDw3UR8A/giphy.gif)
