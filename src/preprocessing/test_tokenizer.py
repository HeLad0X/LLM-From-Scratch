from pathlib import Path
import re
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from GPTConfig import GPTConfig
from preprocessing.BPETokenizer import BPETokenizer


def _load_cached_tokenizer_or_skip():
    cache_dir = Path(getattr(GPTConfig, "token_cache_dir", "./cache_data/bpe_tokenizer"))
    merges = cache_dir / "merges.txt"
    vocab = cache_dir / "vocab.json"
    if not (merges.exists() and vocab.exists()):
        pytest.skip(f"Tokenizer cache missing at {cache_dir}. Run train_bpe.py first.")
    return BPETokenizer.load(str(cache_dir)), cache_dir


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def test_cached_vocab_size_matches_config():
    tok, cache_dir = _load_cached_tokenizer_or_skip()
    cfg_vocab = int(getattr(GPTConfig, "vocab_size", len(tok)))
    model_vocab = int(getattr(GPTConfig, "model_vocab_size", cfg_vocab))
    assert len(tok) == cfg_vocab
    assert len(tok) == model_vocab


@pytest.mark.parametrize(
    "text,expected_snippet",
    [
        ("state-of-the-art models are fun", "state-of-the-art"),
        ('"Hello," she said.', '"hello,"'),
        ('He said "hi there" loudly.', '"hi there"'),
        ("rock-'n'-roll vibes", "rock - 'n' - roll"),
    ],
)
def test_cached_roundtrip_hyphens_and_quotes(text, expected_snippet):
    tok, _ = _load_cached_tokenizer_or_skip()
    decoded = tok.decode(tok.encode(text))
    assert expected_snippet in _normalize(decoded)


def test_cached_handles_numbers_currency_punct():
    tok, _ = _load_cached_tokenizer_or_skip()
    text = "$1,234.50 € and 3.14% -- wow!"
    decoded = tok.decode(tok.encode(text))
    compact = decoded.replace(" ", "")
    assert "$1,234.50".replace(",", "") in compact.replace(",", "")
    assert "3.14%" in compact


def test_cached_special_tokens_present():
    tok, _ = _load_cached_tokenizer_or_skip()
    for special in ("<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SP>"):
        assert special in tok.stoi, f"{special} should be in tokenizer vocab"


def test_cached_encode_words_iter_appends_eos():
    tok, _ = _load_cached_tokenizer_or_skip()
    eos_id = tok.stoi.get("<EOS>")
    assert eos_id is not None
    stream_ids = list(tok.encode_words_iter(["one two", "three four"]))
    assert eos_id in stream_ids
    assert stream_ids[-1] == eos_id


def test_cached_quotes_spacing():
    tok, _ = _load_cached_tokenizer_or_skip()
    text = 'He said, "hello world", then left.'
    decoded = tok.decode(tok.encode(text))
    assert '"hello world"' in decoded


if __name__ == "__main__":
    pytest.main([__file__])
    tok = BPETokenizer.load(GPTConfig.token_cache_dir)
    print(tok.decode(tok.encode("$1,234.50 € and 3.14% -- wow!")))
