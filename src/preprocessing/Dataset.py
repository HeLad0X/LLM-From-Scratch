import torch
from torch.utils.data import IterableDataset
from collections import deque
import random


class GPTIterableDataset(IterableDataset):
    """
    Streams token IDs and yields (x, y) pairs for next-token prediction.
    Assumes the upstream word iterator already includes "<SP>" tokens
    *if* you want explicit spaces (GetDataset.py handles that).
    """

    def __init__(self, word_source, tokenizer, block_size=128, stride=128, repeat=True, shuffle=False, shuffle_buffer=0):
        super().__init__()
        self.word_source = word_source
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.stride = int(stride)
        self.repeat = bool(repeat)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer = int(shuffle_buffer)

        assert 1 <= self.stride <= self.block_size

    def _new_word_iter(self):
        if callable(self.word_source):
            return iter(self.word_source())
        return iter(self.word_source)

    def _token_stream(self):
        # Fast streaming path
        base_iter = self.tokenizer.encode_words_iter(self._new_word_iter())
        if self.shuffle and self.shuffle_buffer > 0:
            yield from self._shuffle_stream(base_iter, self.shuffle_buffer)
        else:
            yield from base_iter

    def _shuffle_stream(self, it, buf_size):
        """
        Streaming shuffle using a fixed-size buffer. When the iterator is
        exhausted, any remaining buffered items are yielded in random order.
        """
        buf = []
        for tok in it:
            buf.append(tok)
            if len(buf) >= buf_size:
                idx = random.randrange(len(buf))
                yield buf.pop(idx)
        random.shuffle(buf)
        for tok in buf:
            yield tok

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
                # Dataset too small to produce one sample.
                if self.repeat:
                    continue
                return

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

            if not self.repeat:
                return

