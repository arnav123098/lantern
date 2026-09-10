class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.has_batch_encoding = hasattr(self.tokenizer, 'encode_batch')

    def encode(self, text):
        return list(self.tokenizer.encode(text))

    def encode_batch(self, batch):
        if not self.has_batch_encoding:
          raise NotImplementedError

        encoded = self.tokenizer.encode_batch(batch)

        return [
            list(x.ids) if hasattr(x, "ids") else list(x)
            for x in encoded
        ]

    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens))

    @property
    def eos_token_id(self):
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def bos_token_id(self):
        return getattr(self.tokenizer, "bos_token_id", None)

    @property
    def pad_token_id(self):
        return getattr(self.tokenizer, "pad_token_id", None)

    def apply_chat_template(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages)

        raise NotImplementedError(
            "Chat templates are not supported by this tokenizer."
        )

    @classmethod
    def from_hf(cls, name: str):
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(name)

        return cls(tokenizer=tokenizer)
    
    @classmethod
    def from_tiktoken(cls, name: str):
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return cls(TikTokenizer(enc))

class TikTokenizer:
    def __init__(self, enc):
        self.enc = enc

        self.eos_token_id = getattr(enc, 'eot_token')

        self.bos_token_id = None

    def encode(self, text): return self.enc.encode(text)

    def decode(self, tokens): return self.enc.decode(tokens)

    def encode_batch(self, batch: list[str]):
        return self.enc.encode_batch(batch)
