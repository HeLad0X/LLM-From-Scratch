# preprocessing/CorpusCleaning.py
import re
import unicodedata
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
REPEAT_PUNCT_RE = re.compile(r"([!?.,\-_=])\1{7,}")
ZW_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


@dataclass
class CleaningConfig:
    # Basic length filters
    min_chars: int = 200
    min_words: int = 30

    # Shape filters
    max_line_len: int = 4000
    max_token_len: int = 200

    # Noise filters
    max_url_ratio: float = 0.05           # url_count / word_count
    min_urls_for_ratio: int = 5           # only apply ratio if at least this many URLs
    max_non_alnum_ratio: float = 0.45     # (not alnum and not space) / total_chars
    max_punct_ratio: float = 0.20         # punctuation / total_chars
    max_replacement_char: int = 3         # '�' occurrences

    # Optional language filter (kept here; actual model is optional)
    enable_language_filter: bool = False
    lang_min_prob: float = 0.80

    # Dedup
    enable_exact_dedup: bool = True
    enable_near_dedup: bool = False
    simhash_bits: int = 64
    simhash_max_hamming: int = 3
    simhash_ngram: int = 4
    simhash_max_chars: int = 4000
    simhash_bucket_bits: int = 16
    simhash_max_candidates: int = 200


class CorpusCleaner:
    """
    Streaming-friendly cleaner.

    It can:
    - normalize Unicode (NFKC) + normalize apostrophes (your current behavior)
    - apply length/shape/noise heuristics
    - exact dedup (hash set) across seen docs in this process
    - near-duplicate detection (SimHash + banded buckets)
    - optional language filter hook (you supply a detector/model if you want)

    Note: "doc" here can be a line, paragraph, or entire file content—
    whatever text unit you feed into `clean_and_check`.
    """

    def __init__(
        self,
        cfg: Optional[CleaningConfig] = None,
        lang_detector=None,   # optional callable: (text)->(lang:str, prob:float)
    ):
        self.cfg = cfg or CleaningConfig()
        self.lang_detector = lang_detector
        self._seen_hashes = set()

        # apostrophe normalization matches your existing normalize_text()
        self._apostrophe_map = str.maketrans({
            "\u2019": "'",  # ’
            "\u2018": "'",  # ‘
            "\u02BC": "'",  # ʼ
        })

        self._simhash_bits = max(8, min(int(self.cfg.simhash_bits), 64))
        bucket_bits = max(4, min(int(self.cfg.simhash_bucket_bits), self._simhash_bits))
        self._simhash_bucket_mask = (1 << bucket_bits) - 1
        self._simhash_bands = [(shift, {}) for shift in range(0, self._simhash_bits, bucket_bits)]

    # ---------- Normalization ----------
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.translate(ZW_CHARS)
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(self._apostrophe_map)
        # Keep newlines for line-based shape checks; collapse later if you want.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse whitespace *within* lines and trim
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ---------- Filters ----------
    def _basic_shape_ok(self, t: str) -> bool:
        if len(t) < self.cfg.min_chars:
            return False

        words = t.split()
        if len(words) < self.cfg.min_words:
            return False

        lines = t.splitlines() or [t]
        if max(len(line) for line in lines) > self.cfg.max_line_len:
            return False

        # cap scan work in case doc is huge
        for w in words[:2000]:
            if len(w) > self.cfg.max_token_len:
                return False

        return True

    def _noise_ok(self, t: str) -> bool:
        # Replacement char / decoding junk
        if t.count("�") > self.cfg.max_replacement_char:
            return False

        # Repeated punctuation blobs
        if REPEAT_PUNCT_RE.search(t):
            return False

        words = t.split()
        wc = max(1, len(words))

        # URL ratio
        urls = URL_RE.findall(t)
        if len(urls) >= self.cfg.min_urls_for_ratio:
            if (len(urls) / wc) > self.cfg.max_url_ratio:
                return False

        n = max(1, len(t))
        non_alnum = 0
        punct = 0
        # `string.punctuation` is ASCII-only; we treat "non-alnum non-space" as symbol-ish.
        for ch in t:
            if not ch.isalnum() and not ch.isspace():
                non_alnum += 1
            # punctuation-ish: treat common ASCII punctuation + some unicode punct
            # This is intentionally light; non_alnum ratio already catches most junk.
            if ch in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""":
                punct += 1

        if (non_alnum / n) > self.cfg.max_non_alnum_ratio:
            return False
        if (punct / n) > self.cfg.max_punct_ratio:
            return False

        return True

    def _language_ok(self, t: str) -> bool:
        if not self.cfg.enable_language_filter:
            return True
        if self.lang_detector is None:
            # If user enabled language filter but didn't provide detector,
            # we fail open (keep) to avoid silently deleting data.
            return True

        sample = t[:2000].replace("\n", " ")
        lang, prob = self.lang_detector(sample)
        return (lang == "en") and (prob >= self.cfg.lang_min_prob)

    # ---------- Dedup ----------
    def _hash_text(self, t: str) -> str:
        # stable and dependency-free
        return hashlib.blake2b(t.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()

    def _is_exact_dup(self, t: str) -> bool:
        if not self.cfg.enable_exact_dedup:
            return False
        h = self._hash_text(t)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def _char_ngrams(self, t: str, n: int):
        if n <= 0:
            return []
        if len(t) <= n:
            return [t]
        return (t[i:i + n] for i in range(len(t) - n + 1))

    def _simhash(self, t: str) -> Optional[int]:
        sample = t[: self.cfg.simhash_max_chars]
        sample = re.sub(r"\s+", " ", sample).strip().lower()
        if not sample:
            return None

        bits = self._simhash_bits
        mask = (1 << bits) - 1
        counts = [0] * bits

        for gram in self._char_ngrams(sample, int(self.cfg.simhash_ngram)):
            h = hashlib.blake2b(gram.encode("utf-8", errors="ignore"), digest_size=8).digest()
            hv = int.from_bytes(h, "big") & mask
            for i in range(bits):
                if (hv >> i) & 1:
                    counts[i] += 1
                else:
                    counts[i] -= 1

        fp = 0
        for i, v in enumerate(counts):
            if v > 0:
                fp |= (1 << i)
        return fp

    def _hamming(self, a: int, b: int) -> int:
        x = a ^ b
        if hasattr(int, "bit_count"):
            return x.bit_count()
        return bin(x).count("1")

    def _is_near_dup(self, t: str) -> bool:
        if not self.cfg.enable_near_dedup:
            return False

        h = self._simhash(t)
        if h is None:
            return False

        max_candidates = max(1, int(self.cfg.simhash_max_candidates))
        checked = 0
        for shift, band in self._simhash_bands:
            key = (h >> shift) & self._simhash_bucket_mask
            for other in band.get(key, []):
                if self._hamming(h, other) <= int(self.cfg.simhash_max_hamming):
                    return True
                checked += 1
                if checked >= max_candidates:
                    break
            if checked >= max_candidates:
                break

        for shift, band in self._simhash_bands:
            key = (h >> shift) & self._simhash_bucket_mask
            band.setdefault(key, []).append(h)
        return False

    # ---------- Public API ----------
    def clean_and_check(self, text: str) -> Tuple[Optional[str], str]:
        """
        Returns:
          (cleaned_text, "ok") if kept
          (None, reason) if dropped
        """
        t = self.normalize(text)
        if not t:
            return None, "empty"

        if not self._basic_shape_ok(t):
            return None, "shape"

        if not self._noise_ok(t):
            return None, "noise"

        if not self._language_ok(t):
            return None, "language"

        if self._is_exact_dup(t):
            return None, "exact_dup"

        if self._is_near_dup(t):
            return None, "near_dup"

        return t, "ok"
