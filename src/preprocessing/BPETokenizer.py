from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
import json, re, unicodedata
import os, sys
from typing import Iterable, List, Dict, Tuple, Optional

# Ensure project root import works when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from GPTConfig import GPTConfig

EOW = "</w>"                 # End-of-word marker used by this simple BPE
SPACE_TOKEN = "<SP>"         # Optional token to represent word boundaries/spaces (only used if present in vocab)

_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    """Unicode normalize + collapse whitespace. Note: whitespace itself is not tokenized by _WORD_RE."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_words(text: str) -> List[str]:
    """Split text into 'words' (alnum runs) and punctuation tokens."""
    text = normalize_text(text)
    if not text:
        return []
    return _WORD_RE.findall(text)


def word_to_symbols(word: str) -> List[str]:
    """Convert a word into base symbols (characters) plus EOW marker."""
    return list(word) + [EOW]


def get_pair_freqs(vocab: Dict[Tuple[str, ...], int]) -> Counter:
    """Count adjacent symbol pair frequencies in the current vocabulary representation."""
    pair_freq = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair_freq[(symbols[i], symbols[i + 1])] += freq
    return pair_freq


def merge_vocab_once(best_pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    """Apply one merge operation to all symbol sequences in vocab."""
    a, b = best_pair
    new_vocab: Dict[Tuple[str, ...], int] = {}
    for symbols, freq in vocab.items():
        i = 0
        merged: List[str] = []
        L = len(symbols)
        while i < L:
            if i < L - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        t = tuple(merged)
        new_vocab[t] = new_vocab.get(t, 0) + freq
    return new_vocab


class BPETokenizer:
    """
    A simple character-level BPE tokenizer.

    Performance improvements vs the original:
      - Precomputes merge ranks once (self._merge_rank) instead of rebuilding per word
      - Adds an LRU cache for encode_word results (massive win for repeated words)
      - Adds encode_words_iter for streaming datasets (avoid regex split per word)
    """

    def __init__(self, specials: Optional[List[str]] = None, cache_size: int = 50000):
        self.specials = specials or GPTConfig.specials
        self.merges: List[Tuple[str, str]] = []
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}

        # Precomputed ranks for merges (lower rank = earlier merge = higher priority)
        self._merge_rank: Dict[Tuple[str, str], int] = {}

        # LRU cache for word -> token ids (word WITHOUT any surrounding whitespace)
        self._cache_size = int(cache_size)
        self._word_cache: "OrderedDict[str, List[int]]" = OrderedDict()

    # -------------------------
    # Training / building vocab
    # -------------------------
    def train(
        self,
        texts: Optional[List[str]] = None,
        words_iter: Optional[Iterable[str]] = None,
        vocab_size: int = GPTConfig.vocab_size,
        min_word_freq: int = GPTConfig.min_word_freq,
        max_merges: Optional[int] = GPTConfig.max_merges,
    ):
        wf = Counter()
        if words_iter is None:
            assert texts is not None, "Provide either texts or words_iter"
            for doc in texts:
                for w in split_words(doc):
                    wf[w] += 1
        else:
            for w in words_iter:
                if not w:
                    continue
                wf[w] += 1

        if min_word_freq > 1:
            wf = Counter({w: c for w, c in wf.items() if c >= min_word_freq})

        # initial vocab = mapping of symbol-tuples -> count
        vocab: Dict[Tuple[str, ...], int] = {tuple(word_to_symbols(w)): c for w, c in wf.items()}

        base_symbols = set()
        for seq in vocab.keys():
            base_symbols.update(seq)

        target_tokens = max(vocab_size - len(self.specials), 0)
        remaining_merges = max(target_tokens - len(base_symbols), 0)
        if max_merges is not None:
            remaining_merges = min(remaining_merges, max_merges)

        merges: List[Tuple[str, str]] = []

        for _ in range(remaining_merges):
            pair_freqs = get_pair_freqs(vocab)
            if not pair_freqs:
                break
            (a, b), cnt = pair_freqs.most_common(1)[0]
            if cnt < 1:
                break
            vocab = merge_vocab_once((a, b), vocab)
            merges.append((a, b))

        # Build final token set from the merged vocab
        tokens = set()
        for seq in vocab.keys():
            tokens.update(seq)

        self.merges = merges
        self.itos = list(self.specials) + sorted(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self._rebuild_merge_rank()
        self._word_cache.clear()

        return {
            "num_merges": len(merges),
            "base_symbols": len(base_symbols),
            "final_vocab_size": len(self.itos),
        }

    def _rebuild_merge_rank(self):
        self._merge_rank = {tuple(p): i for i, p in enumerate(self.merges)}

    # -------------------------
    # Encoding / decoding
    # -------------------------
    def _apply_bpe_to_symbols(self, symbols: List[str]) -> List[str]:
        if not self.merges or len(symbols) < 2:
            return symbols

        s = symbols[:]  # working copy
        merge_rank = self._merge_rank

        while True:
            # Find all adjacent pairs that exist in merge table
            best_rank = None
            best_pair = None

            for i in range(len(s) - 1):
                p = (s[i], s[i + 1])
                r = merge_rank.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p

            if best_pair is None:
                break

            # Merge all occurrences of best_pair in a single pass
            a, b = best_pair
            i = 0
            merged: List[str] = []
            L = len(s)
            while i < L:
                if i < L - 1 and s[i] == a and s[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(s[i])
                    i += 1
            s = merged

        return s

    def _cache_get(self, word: str) -> Optional[List[int]]:
        v = self._word_cache.get(word)
        if v is not None:
            # mark as recently used
            self._word_cache.move_to_end(word, last=True)
        return v

    def _cache_put(self, word: str, ids: List[int]) -> None:
        self._word_cache[word] = ids
        self._word_cache.move_to_end(word, last=True)
        if len(self._word_cache) > self._cache_size:
            self._word_cache.popitem(last=False)

    def encode_word(self, word: str) -> List[int]:
        """Encode a single already-split word/punctuation token."""
        cached = self._cache_get(word)
        if cached is not None:
            return cached

        symbols = word_to_symbols(word)
        merged = self._apply_bpe_to_symbols(symbols)
        unk_id = self.stoi.get("<UNK>", 1)
        ids = [self.stoi.get(t, unk_id) for t in merged]
        self._cache_put(word, ids)
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode a full text string (runs split_words internally)."""
        ids: List[int] = []
        for w in split_words(text):
            ids.extend(self.encode_word(w))
        return ids

    def encode_words_iter(self, word_iter: Iterable[str], add_space_token: bool = False) -> Iterable[int]:
        """
        Stream-encode an iterable of already split words (fast path for datasets).

        If add_space_token=True and SPACE_TOKEN exists in vocab, yields SPACE_TOKEN id between words.
        """
        sp_id = self.stoi.get(SPACE_TOKEN, None) if add_space_token else None
        first = True
        for w in word_iter:
            if not w:
                continue
            if sp_id is not None and not first:
                yield sp_id
            for tid in self.encode_word(w):
                yield tid
            first = False

    def decode(self, ids: List[int]) -> str:
        """
        Decode token ids back to text. This is a simple reconstruction that joins decoded words with spaces.
        If SPACE_TOKEN is present, it is converted to a single space.
        """
        tokens = [self.itos[i] for i in ids]
        words, cur = [], []
        for t in tokens:
            if t == SPACE_TOKEN:
                if cur:
                    words.append("".join(cur))
                    cur = []
                words.append(" ")  # explicit space marker
                continue

            if t == EOW:
                words.append("".join(cur))
                cur = []
            elif t.endswith(EOW):
                base = t[: -len(EOW)]
                cur.append(base)
                words.append("".join(cur))
                cur = []
            else:
                cur.append(t)

        if cur:
            words.append("".join(cur))

        # Collapse consecutive spaces and join
        out = []
        for w in words:
            if w == " ":
                if out and out[-1] != " ":
                    out.append(" ")
            else:
                out.append(w)
        return "".join(out).strip()

    # -------------------------
    # Persistence
    # -------------------------
    def save(self, dirpath: str):
        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")
        with open(p / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos, "specials": self.specials}, f, ensure_ascii=False)

    @classmethod
    def load(cls, dirpath: str) -> "BPETokenizer":
        p = Path(dirpath)
        merges: List[Tuple[str, str]] = []
        with open(p / "merges.txt", "r", encoding="utf-8") as f:
            for line in f:
                a, b = line.rstrip("\n").split(" ")
                merges.append((a, b))
        with open(p / "vocab.json", "r", encoding="utf-8") as f:
            obj = json.load(f)

        tok = cls(specials=obj["specials"])
        tok.merges = merges
        tok.itos = obj["itos"]
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        tok._rebuild_merge_rank()
        tok._word_cache.clear()
        return tok

    def __len__(self):
        return len(self.itos)
