import torch
from torch.utils.data import IterableDataset
from collections import deque


class GPTIterableDataset(IterableDataset):
    """
    Streams token IDs and yields (x, y) pairs for next-token prediction.

    Key improvements vs your previous version:
      - Uses tokenizer.encode_words_iter(...) (fast path) instead of calling tokenizer.encode(...) per word
      - Avoids converting deque -> list twice per sample
      - Optional space token insertion only if tokenizer contains "<SP>"a
    """

    def __init__(self, word_iter, tokenizer, block_size=128, stride=128, add_space_between_words=True):
        super().__init__()
        self.word_iter = word_iter
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.stride = int(stride)

        # NOTE:
        # Your regex-based tokenizer doesn't actually tokenize whitespace.
        # If you want explicit word boundaries, add "<SP>" into your corpus and/or specials and retrain.
        self.add_space_between_words = bool(add_space_between_words)

        assert 1 <= self.stride <= self.block_size

    def _token_stream(self):
        # Fast streaming encoding (no regex splitting per token)
        yield from self.tokenizer.encode_words_iter(
            self.word_iter,
            add_space_token=self.add_space_between_words
        )

    def __iter__(self):
        token_gen = self._token_stream()
        T = self.block_size
        buf = deque(maxlen=T + 1)

        # Prime buffer with T+1 tokens (so we can create x[0:T], y[1:T+1])
        try:
            while len(buf) < (T + 1):
                buf.append(next(token_gen))
        except StopIteration:
            return

        while True:
            arr = list(buf)  # convert once
            x = torch.tensor(arr[:T], dtype=torch.long)
            y = torch.tensor(arr[1:T + 1], dtype=torch.long)
            yield x, y

            # Advance by stride: drop `stride` tokens and append `stride` new tokens
            for _ in range(self.stride):
                try:
                    buf.popleft()
                    buf.append(next(token_gen))
                except StopIteration:
                    return
