import torch
from torch.utils.data import IterableDataset
from collections import deque


class GPTIterableDataset(IterableDataset):
    """
    Streams token IDs and yields (x, y) pairs for next-token prediction.
    Assumes the upstream word iterator already includes "<SP>" tokens
    *if* you want explicit spaces (GetDataset.py handles that).
    """

    def __init__(self, word_iter, tokenizer, block_size=128, stride=128):
        super().__init__()
        self.word_iter = word_iter
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.stride = int(stride)

        assert 1 <= self.stride <= self.block_size

    def _token_stream(self):
        # Fast streaming path
        yield from self.tokenizer.encode_words_iter(self.word_iter)

    def __iter__(self):
        T = self.block_size
        buf = deque(maxlen=T + 1)

        while True:
            token_gen = self._token_stream()
            buf.clear()

            # Prime buffer with T+1 tokens
            try:
                while len(buf) < (T + 1):
                    buf.append(next(token_gen))
            except StopIteration:
                # Dataset too small to even produce one sample.
                # In infinite mode, just try again (or you can raise an error).
                continue

            while True:
                arr = list(buf)
                x = torch.tensor(arr[:T], dtype=torch.long)
                y = torch.tensor(arr[1:T + 1], dtype=torch.long)
                yield x, y

                # Advance by stride
                try:
                    for _ in range(self.stride):
                        buf.popleft()
                        buf.append(next(token_gen))
                except StopIteration:
                    # End of stream → break and restart outer loop with fresh token_gen
                    break

