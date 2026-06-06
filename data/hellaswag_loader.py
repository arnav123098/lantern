import torch
from dataloader import DataLoader
import json
from utils import dotdict

'''
The HellaswagLoader will be used to evaluate the model on the Hellaswag benchmark. We need to download the val dataset of hellaswag.
I've used the batch-wise loading approach as in other dataloaders in the repo.
'''
class HellaswagLoader(DataLoader):
    def __init__(self, config: dotdict):
        super().__init__()

        self.path = 'hellaswag/hellaswag_val.jsonl'

        if not self.downloader.exists(self.path):
            self.downloader.download(
                'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl',
                'hellaswag',
                'hellaswag_val.jsonl'
            )
            
        self.examples = []
        with open(self.downloader.get_path(self.path)) as f:
            for line in f:
                self.examples.append(json.loads(line))
                '''
                The important things we need from the examples are:
                ctx - the context
                labels
                and endings - 4 endings

                Hellaswag evaluates based on the model's answers to multiple choice problems.
                The model has to select the right ending out of 4.

                Since many models such as the GPT2 we've implemented can't solve these directly, we need to make ctx + ending pairs for each endings and get the model's loss on each one. The one with the lowest loss is the one which is associated with the greatest probability i.e. model's answer. We use this logic to evaluate the model on Hellaswag.
                '''

        self.tokenizer = config.tokenizer
        self.batch_size = config.batch_size # 1 batch = 4 sequences
        self.pad_token = config.pad_token

        self.curr_pos = 0

        self.state_keys = ('curr_pos')

    def __len__(self): return len(self.examples)

    def next_batch(self):
        if self.curr_pos >= len(self): return None # completion signal

        end_pos = self.curr_pos + self.batch_size
        buf = self.examples[self.curr_pos:end_pos]

        self.curr_pos = end_pos

        X, M, Y = [], [], []
        for example in buf:
            ctx = example['ctx']
            endings = example['endings']
            label = example['label']

            masks = []
            sequences = []

            '''
            We need to make completion masks for each sequence.
            Let ci be ctx token and ei be ending token.
            If our tokens/sequence looks like [c1, c2, c3, c4, e1, e2, e3],
            then our mask will be [0, 0, 0, 0, 1, 1, 1].
            This tells us where to ignore the loss calculation.
            '''

            ctx_token_len = len(self.tokenizer.encode(ctx))
            for ending in endings:
                seq = self.tokenizer.encode(ctx + ending)
                mask = [0] * ctx_token_len + [1] * (len(seq) - ctx_token_len)
                # encode(ctx + ending) is not necessarily equal to encode(ctx) + encode(ending)

                assert len(mask) == len(seq), "Mask len is not equal to Sequence len"

                sequences.append(seq)
                masks.append(mask)

            X.append(sequences)
            M.append(masks)
            Y.append(label)

        # padding (to make every tensor of equal length i.e. the length of the largest sequence in the batch)
        max_seq_len = max(len(tokens) for seq in X for tokens in seq)
        padded_X = []
        padded_M = []

        for sequences, masks in zip(X, M):
            padded_sequences = []
            padded_masks = []
            for seq, mask in zip(sequences, masks):
                pad_len = max_seq_len - len(seq)
                padded_seq = seq + [self.pad_token] * pad_len
                padded_mask = mask + [0] * pad_len

                assert len(padded_mask) == len(padded_seq), "Padded_ask len is not equal to Padded_Sequence len"

                padded_sequences.append(padded_seq)
                padded_masks.append(padded_mask)

            padded_X.append(padded_sequences)
            padded_M.append(padded_masks)
                
        X, M, Y = torch.tensor(padded_X), torch.tensor(padded_M), torch.tensor(Y)

        return {
            'batch': (X, Y),
            'mask': M 
        }
