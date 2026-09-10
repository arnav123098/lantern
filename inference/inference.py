import torch
import torch.nn.functional as F
from inference.kv_cache import KVCache

# TODO: optimizations for later, cleanups and documentation

class Inference:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.eos_token_id

        model.to(self.device)

    def encode_batch(self, x):
        if self.tokenizer.has_batch_encoding:
            return self.tokenizer.encode_batch(x)

        return [self.tokenizer.encode(xi) for xi in x]

    def pad_tokens(self, tokens: list):
        pad_id = self.tokenizer.pad_token_id

        if pad_id is None:
            print("Tokenizer has no pad token. Using EOS token instead.")
            pad_id = self.eos

        assert pad_id is not None

        max_len = max(len(t) for t in tokens)

        padded = []
        mask = []

        for t in tokens:
            padding = max_len - len(t)

            # left-padding
            padded.append([pad_id] * padding + t)
            mask.append([0] * padding + [1] * len(t))

        return padded, mask

    # This one's essentially the same as the one in Karpathy's Nanochat.
    def sample_next_token(
        self,
        logits,
        rng,
        temperature = 1.0,
        top_k = None
    ): # returns (B, 1)
        assert temperature >= 0.0, "temperature must be non-negative"

        logits = logits[:, -1, :]

        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            vals = vals / temperature
            probs = F.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=rng) # idx in vals
            return idx.gather(1, choice) # map back to actual idx in logits
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1, generator=rng)

    @torch.inference_mode()
    def sample_sequence(
        self,
        x,
        rng,
        attn_mask,
        max_context,
        max_new_tokens,
        temperature,
        top_k,
        kv_cache = None
    ): # non-streaming
        assert max_new_tokens is not None or self.eos is not None, 'No EOS token found. Please specify max_new_tokens.'

        initial_ctx_size = x.size(1)

        new_tokens = 0
        hit_eos_idx = [None] * x.size(0)

        while True:
            if max_new_tokens is not None and new_tokens >= max_new_tokens: break
            if all(idx is not None for idx in hit_eos_idx): break

            ids = x
            current_mask = attn_mask

            if kv_cache is not None:
                if new_tokens != 0:
                    ids = x[:, -1:]
                    current_mask = attn_mask = None # turn off mask

            else:
              if max_context is not None:
                  ids = x[:, -max_context:]

                  if attn_mask is not None:
                      current_mask = [m[-max_context:] for m in attn_mask]

            mask_tensor = torch.tensor(current_mask, dtype=torch.float32, device=self.device)[:, None, None, :].contiguous() if current_mask is not None else None

            logits, _ = self.model.forward(ids, kv_cache=kv_cache, attn_mask=mask_tensor) if mask_tensor is not None else self.model.forward(ids)
            next = self.sample_next_token(logits, rng, temperature, top_k)

            x = torch.cat((x, next), dim=1)

            # TODO: optimize later
            hit_eos = next[:, 0] == self.eos if self.eos is not None else [False] * next.size(0)
            hit_eos_idx = [initial_ctx_size + new_tokens if (idx is None and hit) else idx for hit, idx in zip(hit_eos, hit_eos_idx)]

            new_tokens += 1
            if current_mask is not None:
                attn_mask = [m + [1] for m in attn_mask]

        seq_len = x.size(1)
        finished_mask = [
            [1] * (idx + 1) + [0] * (seq_len - (idx + 1))
            if idx is not None else [1] * seq_len
            for idx in hit_eos_idx
        ]

        return x, finished_mask

    def generate(
            self,
            prompt: str | list[str],
            *,
            use_kv_cache: bool = False,
            max_context: int | None = None,
            max_new_tokens: int | None = None,
            seed: int = 69,
            temperature = 1.0,
            top_k = None
        ):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)

        if isinstance(prompt, str):
            tokens = [self.tokenizer.encode(prompt)]
            batch_size = 1
        else:
            tokens = self.encode_batch(prompt)
            batch_size = len(prompt)

        kv_cache = KVCache(
            batch_size,
            self.model.config.n_head,
            self.model.config.block_size,
            self.model.config.n_embd // self.model.config.n_head,
            self.model.config.n_layer,
            self.device,
            torch.float32
        ) if use_kv_cache else None

        if len(tokens) > 1:
            tokens, mask = self.pad_tokens(tokens)
            if use_kv_cache:
                mask = None
        else:
            mask = None

        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens.to(self.device)

        batch, finished_mask = self.sample_sequence(x, rng, mask, max_context, max_new_tokens, temperature, top_k, kv_cache = kv_cache)

        for sample, f_mask in zip(batch, finished_mask):
            decoded = self.tokenizer.decode(sample.tolist()[:sum(f_mask)])
            print(">", decoded)
