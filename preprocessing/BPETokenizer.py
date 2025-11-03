from collections import Counter
from pathlib import Path
import json, re, unicodedata

EOW = "</w>"  

_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def normalize_text(s: str):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_words(text: str):
    text = normalize_text(text)
    if not text: return []
    return _WORD_RE.findall(text)

def word_to_symbols(word: str):
    return list(word) + [EOW]

def get_pair_freqs(vocab: dict[tuple, int]) -> Counter:
    pair_freq = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair_freq[(symbols[i], symbols[i+1])] += freq
    return pair_freq

def merge_vocab_once(best_pair: tuple[str, str], vocab: dict[tuple, int]) -> dict:
    a, b = best_pair
    new_vocab: dict[tuple, int] = {}
    for symbols, freq in vocab.items():
        i = 0
        merged = []
        L = len(symbols)
        while i < L:
            if i < L - 1 and symbols[i] == a and symbols[i+1] == b:
                merged.append(a + b)  
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        t = tuple(merged)
        new_vocab[t] = new_vocab.get(t, 0) + freq
    return new_vocab

class BPETokenizer:
    def __init__(self, specials=None):
        self.specials = specials or ["<PAD>", "<UNK>"]
        self.merges: list[tuple[str, str]] = []  
        self.itos: list[str] = []  
        self.stoi: dict[str, int] = {}        

    def train(self,
              texts: list[str] | None = None,
              words_iter=None,
              vocab_size: int = 4000,
              min_word_freq: int = 1,
              max_merges: int | None = None):

        wf = Counter()
        if words_iter is None:
            assert texts is not None, "Provide either texts or words_iter"
            for doc in texts:
                for w in split_words(doc):
                    wf[w] += 1
        else:
            for w in words_iter:
                if not w: continue
                wf[w] += 1

        if min_word_freq > 1:
            wf = Counter({w: c for w, c in wf.items() if c >= min_word_freq})

        vocab = {tuple(word_to_symbols(w)): c for w, c in wf.items()}

        base_symbols = set()
        for seq in vocab.keys():
            base_symbols.update(seq)

        target_tokens = max(vocab_size - len(self.specials), 0)
        remaining_merges = max(target_tokens - len(base_symbols), 0)
        if max_merges is not None:
            remaining_merges = min(remaining_merges, max_merges)

        merges: list[tuple[str, str]] = []

        for _ in range(remaining_merges):
            pair_freqs = get_pair_freqs(vocab)
            if not pair_freqs:
                break
            (a, b), cnt = pair_freqs.most_common(1)[0]
            if cnt < 1:
                break
            vocab = merge_vocab_once((a, b), vocab)
            merges.append((a, b))

        tokens = set()
        for seq in vocab.keys():
            tokens.update(seq)

        itos = list(self.specials) + sorted(tokens)
        stoi = {t: i for i, t in enumerate(itos)}

        self.merges = merges
        self.itos = itos
        self.stoi = stoi

        return {
            "num_merges": len(merges),
            "base_symbols": len(base_symbols),
            "final_vocab_size": len(itos)
        }

    def _apply_bpe_to_symbols(self, symbols: list[str]) -> list[str]:
        if not self.merges or len(symbols) < 2:
            return symbols

        merge_rank = {tuple(p): i for i, p in enumerate(self.merges)}

        s = symbols[:]
        while True:
            pairs = [(s[i], s[i+1]) for i in range(len(s) - 1)]
            ranked = [(merge_rank.get(p, None), p) for p in pairs]
            ranked = [r for r in ranked if r[0] is not None]
            if not ranked:
                break

            _, best = min(ranked, key=lambda x: x[0])

            i = 0
            merged = []
            L = len(s)
            while i < L:
                if i < L - 1 and (s[i], s[i+1]) == best:
                    merged.append(s[i] + s[i+1])
                    i += 2
                else:
                    merged.append(s[i])
                    i += 1
            s = merged

        return s

    def encode_word(self, word: str) -> list[int]:
        symbols = word_to_symbols(word)
        merged = self._apply_bpe_to_symbols(symbols)
        unk_id = self.stoi.get("<UNK>", 1)
        return [self.stoi.get(t, unk_id) for t in merged]

    def encode(self, text: str) -> list[int]:
        ids = []
        for w in split_words(text):
            ids.extend(self.encode_word(w))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.itos[i] for i in ids]
        words, cur = [], []
        for t in tokens:
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
        return " ".join(w for w in words if w)

    def save(self, dirpath: str):
        p = Path(dirpath); p.mkdir(parents=True, exist_ok=True)
        with open(p / "merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")
        with open(p / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos, "specials": self.specials}, f, ensure_ascii=False)

    @classmethod
    def load(cls, dirpath: str) -> "BPETokenizer":
        p = Path(dirpath)
        merges = []
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
        return tok

    def __len__(self):
        return len(self.itos)
