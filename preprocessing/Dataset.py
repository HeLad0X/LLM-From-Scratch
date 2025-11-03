import torch
from torch.utils.data import IterableDataset
from collections import deque

class GPTIterableDataset(IterableDataset):
    def __init__(self, word_iter, tokenizer, block_size=128, stride=128, add_space_between_words=True):
        super().__init__()
        self.word_iter = word_iter
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.stride = int(stride)
        self.add_space_between_words = add_space_between_words

        assert 1 <= self.stride <= self.block_size

    def _token_stream(self):
        first = True
        for w in self.word_iter:
            if self.add_space_between_words and not first:
                # Add a separating space between words, then encode word
                for tid in self.tokenizer.encode(" " + w):
                    yield tid
            else:
                for tid in self.tokenizer.encode(w):
                    yield tid
                first = False

    def __iter__(self):
        token_gen = self._token_stream()
        buf = deque(maxlen=self.block_size + 1)

        # Prime the buffer
        try:
            while len(buf) < (self.block_size + 1):
                buf.append(next(token_gen))
        except StopIteration:
            # Not enough tokens to form even one sample
            return

        while True:
            # Emit one sample
            x = torch.tensor(list(buf)[:self.block_size], dtype=torch.long)
            y = torch.tensor(list(buf)[1:self.block_size+1], dtype=torch.long)
            yield x, y

            # Advance by stride
            for _ in range(self.stride):
                if buf:
                    buf.popleft()
                try:
                    buf.append(next(token_gen))
                except StopIteration:
                    # If we can’t refill to full length, we’re done
                    if len(buf) < (self.block_size + 1):
                        return
