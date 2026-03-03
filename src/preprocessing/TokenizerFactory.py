from __future__ import annotations

from pathlib import Path

from GPTConfig import GPTConfig
from preprocessing.BPETokenizer import BPETokenizer, split_words


def load_tokenizer():
    cache_dir = Path(getattr(GPTConfig, "token_cache_dir", "./cache_data/bpe_tokenizer"))
    merges_path = cache_dir / "merges.txt"
    vocab_path = cache_dir / "vocab.json"
    if not (merges_path.exists() and vocab_path.exists()):
        raise FileNotFoundError(
            f"BPE tokenizer not found in {cache_dir}. "
            "Run train_bpe.py or load_or_build_tokenizer(...) with a corpus iterator to build it."
        )
    tok = BPETokenizer.load(str(cache_dir))
    validate_tokenizer_config(tok)
    return tok


def load_or_build_tokenizer(train_iter_factory=None):
    cache_dir = Path(getattr(GPTConfig, "token_cache_dir", "./cache_data/bpe_tokenizer"))
    merges_path = cache_dir / "merges.txt"
    vocab_path = cache_dir / "vocab.json"

    if merges_path.exists() and vocab_path.exists():
        tok = BPETokenizer.load(str(cache_dir))
        validate_tokenizer_config(tok)
        return tok

    if train_iter_factory is None:
        raise ValueError(
            "Custom BPE tokenizer cache is missing and no train_iter_factory was provided "
            "to build it. Pass a corpus iterator factory to load_or_build_tokenizer()."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer(
        specials=getattr(GPTConfig, "specials", None),
        cache_size=getattr(GPTConfig, "cache_size", 50000),
    )

    # Streaming word iterator over cleaned text
    def word_iter():
        for doc in train_iter_factory():
            if not doc:
                continue
            for w in split_words(doc):
                yield w

    tok.train(
        words_iter=word_iter(),
        vocab_size=int(getattr(GPTConfig, "vocab_size", 32000)),
        min_word_freq=int(getattr(GPTConfig, "min_word_freq", 1)),
        max_merges=getattr(GPTConfig, "max_merges", None),
        log_dir=str(getattr(GPTConfig, "log_dir", "./logs")),
        log_every_seconds=int(getattr(GPTConfig, "log_interval", 60)),
    )

    # Persist immediately, then reload from disk so subsequent users see the saved copy.
    tok.save(str(cache_dir))
    tok = BPETokenizer.load(str(cache_dir))

    validate_tokenizer_config(tok)
    return tok


def validate_tokenizer_config(tok) -> None:
    tok_vocab = len(tok)
    cfg_tok_vocab = int(getattr(GPTConfig, "vocab_size", tok_vocab) or tok_vocab)
    cfg_model_vocab = int(getattr(GPTConfig, "model_vocab_size", cfg_tok_vocab) or cfg_tok_vocab)

    if tok_vocab != cfg_tok_vocab:
        raise ValueError(
            f"Tokenizer vocab mismatch: tokenizer has {tok_vocab}, "
            f"but tokenizer.vocab_size in config is {cfg_tok_vocab}."
        )
    if cfg_model_vocab != cfg_tok_vocab:
        raise ValueError(
            f"Model/config vocab mismatch: model.vocab_size={cfg_model_vocab}, "
            f"tokenizer.vocab_size={cfg_tok_vocab}."
        )
