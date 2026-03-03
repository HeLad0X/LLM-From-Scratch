from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
import json, re, unicodedata
import os, sys
import time, logging, datetime
from typing import Iterable, List, Dict, Tuple, Optional

# Ensure project root import works when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from GPTConfig import GPTConfig

EOW = "</w>"  # End-of-word marker used by this simple BPE

_WORD_RE = re.compile(
    r"(?:\w+(?:'\w+)*)"      # words with apostrophe contractions (don't, I've)
    r"|--"                   # double dash
    r"|-"                    # hyphen-minus (after normalization, covers many dashes)
    r'|"'                    # double quote
    r"|[^\w\s]",             # any other single non-word non-space char
    flags=re.UNICODE
)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)

    # Normalize curly quotes/apostrophes to ASCII
    s = s.translate(str.maketrans({
        "\u2019": "'",  # ’
        "\u2018": "'",  # ‘
        "\u02BC": "'",  # ʼ

        "\u201C": '"',  # “
        "\u201D": '"',  # ”
        "\u201E": '"',  # „
        "\u00AB": '"',  # «
        "\u00BB": '"',  # »

        "\u2010": "-",  # ‐
        "\u2011": "-",  # -
        "\u2012": "-",  # ‒
        "\u2013": "-",  # –
        "\u2014": "-",  # —
        "\u2212": "-",  # −
    }))

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
    pair_freq = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair_freq[(symbols[i], symbols[i + 1])] += freq
    return pair_freq


def merge_vocab_once(best_pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
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
    def __init__(self, specials: Optional[List[str]] = None, cache_size: int = GPTConfig.cache_size):
        # SINGLE source of truth for specials
        self.specials = list(specials) if specials is not None else list(GPTConfig.specials)

        # If you want explicit spaces, include "<SP>" inside GPTConfig.specials
        self.space_token = "<SP>" if "<SP>" in self.specials else None

        self.merges: List[Tuple[str, str]] = []
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}

        self._merge_rank: Dict[Tuple[str, str], int] = {}

        self._cache_size = int(cache_size)
        self._word_cache: "OrderedDict[str, List[int]]" = OrderedDict()

    def _get_train_logger(self, log_dir: str = "./logs") -> logging.Logger:
        logger_name = f"BPETokenizer.train.{id(self)}"
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            return logger

        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"bpe_train_{ts}.log")

        logger.setLevel(logging.INFO)
        logger.propagate = False

        fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        logger.addHandler(ch)
        logger.addHandler(fh)

        logger.info(f"[BPE] Training log file: {log_path}")
        return logger

    def train(
        self,
        texts: Optional[List[str]] = None,
        words_iter: Optional[Iterable[str]] = None,
        vocab_size: int = GPTConfig.vocab_size,
        min_word_freq: int = GPTConfig.min_word_freq,
        max_merges: Optional[int] = GPTConfig.max_merges,
        log_dir: str = GPTConfig.log_dir,
        log_every_seconds: int = GPTConfig.log_interval,
    ):
        logger = self._get_train_logger(log_dir=log_dir)
        start_t = time.monotonic()
        last_log_t = start_t

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
                # Important: if your corpus iterator emits "<SP>", we typically
                # do NOT want to count it as a mergeable "word".
                if w in self.specials:
                    continue
                wf[w] += 1

        if min_word_freq > 1:
            wf = Counter({w: c for w, c in wf.items() if c >= min_word_freq})

        vocab: Dict[Tuple[str, ...], int] = {tuple(word_to_symbols(w)): c for w, c in wf.items()}

        base_symbols = set()
        for seq in vocab.keys():
            base_symbols.update(seq)

        target_tokens = max(vocab_size - len(self.specials), 0)
        remaining_merges = max(target_tokens - len(base_symbols), 0)
        if max_merges is not None:
            remaining_merges = min(remaining_merges, max_merges)

        logger.info(f"[BPE] Starting training: target vocab_size={vocab_size}, min_word_freq={min_word_freq}, max_merges={max_merges}")
        logger.info(f"[BPE] Base symbols={len(base_symbols)}, planned merges={remaining_merges}, unique_words={len(wf)}")

        merges: List[Tuple[str, str]] = []
        for i in range(remaining_merges):
            pair_freqs = get_pair_freqs(vocab)
            if not pair_freqs:
                logger.info("[BPE] Stopping early: no more pairs to merge.")
                break

            (a, b), cnt = pair_freqs.most_common(1)[0]
            if cnt < 1:
                logger.info("[BPE] Stopping early: best pair frequency < 1.")
                break

            vocab = merge_vocab_once((a, b), vocab)
            merges.append((a, b))

            now_t = time.monotonic()
            if (now_t - last_log_t) >= max(int(log_every_seconds), 1):
                done = i + 1
                progress = (done / remaining_merges * 100.0) if remaining_merges else 100.0
                elapsed = now_t - start_t
                avg = elapsed / done if done else 0.0
                eta = avg * (remaining_merges - done) if remaining_merges else 0.0
                logger.info(
                    f"[BPE] {done}/{remaining_merges} merges ({progress:.2f}%) | "
                    f"elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m | best_pair_freq={cnt}"
                )
                last_log_t = now_t

        tokens = set()
        for seq in vocab.keys():
            tokens.update(seq)

        self.merges = merges
        self.itos = list(self.specials) + sorted(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        # If the corpus has fewer base symbols/merges than target vocab,
        # pad with synthetic tokens to hit the requested vocab size so config stays consistent.
        target_vocab = int(vocab_size)
        deficit = max(0, target_vocab - len(self.itos))
        if deficit > 0:
            for i in range(deficit):
                t = f"<EXTRA_{i}>"
                self.stoi[t] = len(self.itos)
                self.itos.append(t)
            logger.info(f"[BPE] Added {deficit} synthetic tokens to reach vocab_size={target_vocab}")

        self._rebuild_merge_rank()
        self._word_cache.clear()

        end_t = time.monotonic()
        logger.info(
            f"[BPE] Training finished: merges_done={len(merges)}/{remaining_merges}, "
            f"final_vocab_size={len(self.itos)}, elapsed={(end_t-start_t)/60:.1f}m"
        )
        return {"num_merges": len(merges), "base_symbols": len(base_symbols), "final_vocab_size": len(self.itos)}

    def _rebuild_merge_rank(self):
        self._merge_rank = {tuple(p): i for i, p in enumerate(self.merges)}

    def _apply_bpe_to_symbols(self, symbols: List[str]) -> List[str]:
        if not self.merges or len(symbols) < 2:
            return symbols

        s = symbols[:]
        merge_rank = self._merge_rank

        while True:
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
            self._word_cache.move_to_end(word, last=True)
        return v

    def _cache_put(self, word: str, ids: List[int]) -> None:
        self._word_cache[word] = ids
        self._word_cache.move_to_end(word, last=True)
        if len(self._word_cache) > self._cache_size:
            self._word_cache.popitem(last=False)

    def encode_word(self, tok: str) -> List[int]:
        """
        Encode a single already-split token.
        - Specials (e.g. <SP>, <BOS>, ...) are atomic.
        - Normal tokens go through char-BPE with EOW.
        """
        if tok in self.specials:
            tid = self.stoi.get(tok)
            if tid is None:
                tid = self.stoi.get("<UNK>", 1)
            return [tid]

        cached = self._cache_get(tok)
        if cached is not None:
            return cached

        symbols = word_to_symbols(tok)
        merged = self._apply_bpe_to_symbols(symbols)
        unk_id = self.stoi.get("<UNK>", 1)
        ids = [self.stoi.get(t, unk_id) for t in merged]
        self._cache_put(tok, ids)
        return ids

    def encode(self, text: str) -> List[int]:
        """
        Encode raw text. If <SP> exists in specials, we emit it between *words*.
        """
        ids: List[int] = []
        first = True
        for tok in split_words(text):
            if not tok:
                continue

            is_word = tok[0].isalnum() or tok[0] == "_"
            if (not first) and is_word and self.space_token is not None:
                ids.extend(self.encode_word(self.space_token))

            ids.extend(self.encode_word(tok))
            first = False

        return ids

    def encode_words_iter(self, word_iter):
        """
        Streaming encoder for datasets.
        Accepts either pre-tokenized words/specials or raw text chunks.
        - If item is a known special token, emit it directly.
        - If item contains whitespace (likely a document/chunk), run full encode() and append <EOS>.
        - Otherwise treat it as a single token/word.
        """
        eos = self.stoi.get("<EOS>")
        specials_set = set(self.specials)

        for item in word_iter:
            if not item:
                continue

            if isinstance(item, str) and item in specials_set:
                yield from self.encode_word(item)
                continue

            if isinstance(item, str) and any(ch.isspace() for ch in item):
                yield from self.encode(item)
                if eos is not None:
                    yield eos
                continue

            # Fallback: treat as pre-split word
            yield from self.encode_word(str(item))

    def _detokenize_text(self, s: str) -> str:
        # Normalize whitespace first.
        s = re.sub(r"[ \t]+", " ", s).strip()

        # Remove spaces before common closing punctuation.
        s = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", s)

        # Remove spaces after common opening punctuation.
        s = re.sub(r"([(\[\{])\s+", r"\1", s)

        # Currency formatting: "$ 10" -> "$10"
        s = re.sub(r"([$€£¥])\s+(\d)", r"\1\2", s)

        # Decimal and grouped numbers: "1. 99" -> "1.99", "1, 000" -> "1,000"
        s = re.sub(r"(\d)\s*([.,])\s*(\d)", r"\1\2\3", s)

        # Hyphenated compounds: "two - year - old" or "two- year" -> "two-year-old"
        # Run twice to catch chains like a - b - c (first pass: a-b - c, second: a-b-c)
        for _ in range(3):
            s = re.sub(r"([A-Za-z0-9])\s*-\s*([A-Za-z0-9])", r"\1-\2", s)
            
        for _ in range(3):
            s = re.sub(r"([A-Za-z0-9])\s*-\s*([A-Za-z0-9])", r"\1-\2", s)

        # English contractions / possessives: "don ' t" -> "don't", "it ' s" -> "it's"
        s = re.sub(r"([A-Za-z])\s+'\s*([A-Za-z])", r"\1'\2", s)
        s = re.sub(r"([A-Za-z])\s*'\s+([A-Za-z])", r"\1'\2", s)

        # ---- Double-quote spacing ----
        # Process quotes as pairs: strip whitespace immediately inside "..." 
        # e.g. ' " hello " ' -> '"hello"', 'said " hi " and' -> 'said "hi" and'
        s = re.sub(r'"(\s*)(.*?)(\s*)"', lambda m: '"' + m.group(2).strip() + '"', s)

        # ---- Single-quote spacing (quotation marks only, not contractions) ----
        # Only fix single quotes that are clearly acting as quotation marks.
        # Opening single quote: preceded by space/start, followed by word (with possible space)
        s = re.sub(r"(^|\s)'\s*([A-Za-z0-9])", lambda m: m.group(1) + "'" + m.group(2), s)
        # Closing single quote: word followed by space then quote then space/end
        s = re.sub(r"([A-Za-z0-9])\s+'(\s|$)", lambda m: m.group(1) + "'" + m.group(2), s)

        # Collapse spaces again after substitutions.
        s = re.sub(r"[ \t]+", " ", s).strip()
        return s

    def decode(self, ids: List[int]) -> str:
        tokens = [self.itos[i] for i in ids]
        out_chunks: List[str] = []
        cur: List[str] = []

        for t in tokens:
            # Space token boundary
            if self.space_token is not None and t == self.space_token:
                if cur:
                    out_chunks.append("".join(cur))
                    cur = []
                out_chunks.append(" ")
                continue

            # Token is exactly EOW (bare end-of-word marker — rare but possible)
            if t == EOW:
                if cur:
                    out_chunks.append("".join(cur))
                    cur = []
                else:
                    # bare </w> with nothing accumulated — emit nothing
                    pass
                continue

            # Token ends with EOW — it's a complete word-final subword
            if t.endswith(EOW):
                cur.append(t[:-len(EOW)])
                out_chunks.append("".join(cur))
                cur = []
                out_chunks.append(" ")  # word boundary → natural space
                continue

            # Token contains EOW internally (shouldn't happen in well-formed BPE,
            # but guard against it anyway)
            if EOW in t:
                parts = t.split(EOW)
                cur.append(parts[0])
                out_chunks.append("".join(cur))
                cur = []
                out_chunks.append(" ")
                # anything after EOW is a new fragment
                if parts[1]:
                    cur.append(parts[1])
                continue

            # Plain subword fragment — accumulate
            cur.append(t)

        # Flush any remaining fragment (incomplete word at end of sequence)
        if cur:
            out_chunks.append("".join(cur))

        # Join all chunks — at this point words are space-separated naturally
        # because we emitted " " after every EOW token.
        s = "".join(out_chunks).strip()

        # Apply post-processing to fix punctuation spacing artifacts.
        return self._detokenize_text(s)

    def decode_tensor(self, ids_tensor) -> str:
        if hasattr(ids_tensor, "detach"):
            ids_tensor = ids_tensor.detach().cpu()
        return self.decode(ids_tensor.tolist())

    def decode_batch(self, ids_tensor) -> list[str]:
        if hasattr(ids_tensor, "detach"):
            ids_tensor = ids_tensor.detach().cpu()
        return [self.decode(row.tolist()) for row in ids_tensor]

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
                a, b = line.rstrip("\n").rsplit(" ", 1)
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
