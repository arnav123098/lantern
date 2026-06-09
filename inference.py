import torch

class Generate:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, num_return_sequences=3, prompt="what is a large language model?"):
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x = tokens.to(self.model.device)
        
        for i in range(num_return_sequences):
            tokens = self.model.generate(x)[i, :].tolist()
            decoded = self.tokenizer.decode(tokens)
            print(">", decoded)
