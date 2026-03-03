# GetDataset.py (updated)
import os
import sys
import functools
from torch.utils.data import DataLoader

# Ensure project root import works when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from preprocessing.Dataset import GPTIterableDataset
from preprocessing.CleanCorpus import CorpusCleaner, CleaningConfig
from preprocessing.TokenizerFactory import load_or_build_tokenizer
from GPTConfig import GPTConfig

def get_train_val_loaders():
    train_dir = f"{GPTConfig.data_dir}/train"
    val_dir   = f"{GPTConfig.data_dir}/eval"

    if not os.path.isdir(train_dir):
        train_dir = GPTConfig.data_dir

    print("Train directory:", train_dir)

    train_iter_data = get_corpus_iter(train_dir)
    tok = load_or_build_tokenizer(train_iter_factory=train_iter_data)
    train_ds = GPTIterableDataset(
        train_iter_data, tok,
        block_size=GPTConfig.block_size,
        stride=getattr(GPTConfig, "stride", GPTConfig.block_size),
        repeat=True,
        shuffle=getattr(GPTConfig, "shuffle", False),
        shuffle_buffer=getattr(GPTConfig, "shuffle_buffer", 0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=GPTConfig.batch_size,
        num_workers=getattr(GPTConfig, "num_workers", 4),
        pin_memory=getattr(GPTConfig, "pin_memory", False),
        shuffle=False,
    )

    val_loader = None
    if os.path.isdir(val_dir):
        print("Val directory:", val_dir)
        val_iter_data = get_corpus_iter(val_dir)
        val_ds = GPTIterableDataset(
            val_iter_data, tok,
            block_size=GPTConfig.block_size,
            stride=getattr(GPTConfig, "stride", GPTConfig.block_size),
            repeat=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=GPTConfig.batch_size,
            num_workers=getattr(GPTConfig, "num_workers", 0),
            pin_memory=getattr(GPTConfig, "pin_memory", False),
            shuffle=False,
        )

    return train_loader, val_loader


def _make_cleaner() -> CorpusCleaner:
    cfg = CleaningConfig(
        min_chars=getattr(GPTConfig, "clean_min_chars", 200),
        min_words=getattr(GPTConfig, "clean_min_words", 30),
        max_line_len=getattr(GPTConfig, "clean_max_line_len", 4000),
        max_token_len=getattr(GPTConfig, "clean_max_token_len", 200),
        max_url_ratio=getattr(GPTConfig, "clean_max_url_ratio", 0.05),
        min_urls_for_ratio=getattr(GPTConfig, "clean_min_urls_for_ratio", 5),
        max_non_alnum_ratio=getattr(GPTConfig, "clean_max_non_alnum_ratio", 0.45),
        max_punct_ratio=getattr(GPTConfig, "clean_max_punct_ratio", 0.20),
        max_replacement_char=getattr(GPTConfig, "clean_max_replacement_char", 3),
        enable_language_filter=getattr(GPTConfig, "clean_enable_language_filter", False),
        lang_min_prob=getattr(GPTConfig, "clean_lang_min_prob", 0.80),
        enable_exact_dedup=getattr(GPTConfig, "clean_enable_exact_dedup", True),
        enable_near_dedup=getattr(GPTConfig, "clean_enable_near_dedup", False),
        simhash_bits=getattr(GPTConfig, "clean_simhash_bits", 64),
        simhash_max_hamming=getattr(GPTConfig, "clean_simhash_max_hamming", 3),
        simhash_ngram=getattr(GPTConfig, "clean_simhash_ngram", 4),
        simhash_max_chars=getattr(GPTConfig, "clean_simhash_max_chars", 4000),
        simhash_bucket_bits=getattr(GPTConfig, "clean_simhash_bucket_bits", 16),
        simhash_max_candidates=getattr(GPTConfig, "clean_simhash_max_candidates", 200),
    )
    return CorpusCleaner(cfg=cfg, lang_detector=None)


def iter_clean_texts_from_dir(data_dir: str, max_chunk_chars: int = 4000):
    """
    Streams cleaned text in larger chunks to avoid excessive <|endoftext|> frequency.
    Lines are cleaned individually, then grouped until max_chunk_chars is reached.
    """
    cleaner = _make_cleaner()

    dropped = {"empty": 0, "shape": 0, "noise": 0, "language": 0, "exact_dup": 0, "near_dup": 0}
    kept = 0

    buf = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            yield " ".join(buf)
        buf = []
        buf_len = 0

    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        files.sort()
        for fname in files:
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    cleaned, reason = cleaner.clean_and_check(line)
                    if cleaned is None:
                        if reason in dropped:
                            dropped[reason] += 1
                        continue
                    kept += 1

                    # Accumulate into chunks to reduce EOT token frequency.
                    if buf_len + len(cleaned) > max_chunk_chars:
                        for chunk in flush():
                            yield chunk
                    buf.append(cleaned)
                    buf_len += len(cleaned) + 1

            # flush at file boundary
            for chunk in flush():
                yield chunk

    # final flush
    for chunk in flush():
        yield chunk

    # If you want visibility, uncomment:
    # print(f"[Cleaner] kept={kept}, dropped={dropped}")


def get_corpus_iter(data_path: str):
    """Factory so datasets can create a fresh iterator each time."""
    return functools.partial(iter_clean_texts_from_dir, data_path)


def get_dataset_iter(split="train"):
    data_path = os.path.join(GPTConfig.data_dir, split)
    corpus_factory = get_corpus_iter(data_path)

    tokenizer = load_or_build_tokenizer(train_iter_factory=corpus_factory)

    print(f"Finished Loading/Training Tokenizer. Vocab size: {len(tokenizer)}. Creating dataset...")

    dataset = GPTIterableDataset(
        corpus_factory,
        tokenizer,
        block_size=GPTConfig.block_size,
        stride=GPTConfig.stride,
        repeat=(split == "train"),
    )

    batch_size = getattr(GPTConfig, "batch_size", 8)
    num_workers = getattr(GPTConfig, "num_workers", 0)
    pin_memory = getattr(GPTConfig, "pin_memory", True)

    print("Created dataset. Creating DataLoader...")
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return data_loader


if __name__ == "__main__":
    data_loader_iter = iter(get_dataset_iter())
    next_batch = next(data_loader_iter)
    print(next_batch)
