# GetDataset.py (updated)
import os
import sys
import re
import unicodedata
from torch.utils.data import DataLoader

# Ensure project root import works when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from preprocessing.BPETokenizer import BPETokenizer
from preprocessing.Dataset import GPTIterableDataset
from preprocessing.CleanCorpus import CorpusCleaner, CleaningConfig
from GPTConfig import GPTConfig

WORD_RE = re.compile(r"\w+(?:'\w+)*|[^\w\s]", flags=re.UNICODE)

def get_train_val_loaders():
    tok = BPETokenizer.load(GPTConfig.token_cache_dir)

    train_dir = f"{GPTConfig.data_dir}/train"
    val_dir   = f"{GPTConfig.data_dir}/eval"

    if not os.path.isdir(train_dir):
        train_dir = GPTConfig.data_dir

    print("Train directory:", train_dir)

    train_iter_data = iter_words_from_dir(train_dir)
    train_ds = GPTIterableDataset(
        train_iter_data, tok,
        block_size=GPTConfig.block_size,
        stride=getattr(GPTConfig, "stride", GPTConfig.block_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=GPTConfig.batch_size,
        num_workers=getattr(GPTConfig, "num_workers", 0),
        pin_memory=getattr(GPTConfig, "pin_memory", False),
        shuffle=False,
    )

    val_loader = None
    if os.path.isdir(val_dir):
        print("Val directory:", val_dir)
        val_iter_data = iter_words_from_dir(val_dir)
        val_ds = GPTIterableDataset(
            val_iter_data, tok,
            block_size=GPTConfig.block_size,
            stride=getattr(GPTConfig, "stride", GPTConfig.block_size),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=GPTConfig.batch_size,
            num_workers=getattr(GPTConfig, "num_workers", 0),
            pin_memory=getattr(GPTConfig, "pin_memory", False),
            shuffle=False,
        )

    return train_loader, val_loader


def iter_words_from_dir(data_dir: str):
    """
    Yields a stream of tokens from cleaned text lines:
      - word tokens and punctuation tokens from WORD_RE
      - optionally emits explicit space token "<SP>" between words (if present in GPTConfig.specials)
    """
    space_tok = "<SP>" if "<SP>" in getattr(GPTConfig, "specials", []) else None

    # Pull optional cleaning knobs from GPTConfig if you add them; otherwise defaults apply.
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

    # If you later add fastText/langid, pass a detector here.
    cleaner = CorpusCleaner(cfg=cfg, lang_detector=None)

    # Optional: basic counters for debugging (won't spam unless you print)
    dropped = {"empty": 0, "shape": 0, "noise": 0, "language": 0, "exact_dup": 0, "near_dup": 0}
    kept = 0

    for root, _, files in os.walk(data_dir):
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

                    first = True
                    for tok in WORD_RE.findall(cleaned):
                        if not tok:
                            continue

                        is_word = tok[0].isalnum() or tok[0] == "_"

                        if (not first) and is_word and space_tok is not None:
                            yield space_tok
                            yield tok
                        else:
                            yield tok

                        first = False

    # If you want visibility, uncomment:
    # print(f"[Cleaner] kept={kept}, dropped={dropped}")


def get_corpus_iter(data_path: str):
    """Factory so datasets can create a fresh iterator each time."""
    def factory():
        return iter_words_from_dir(data_path)
    return factory


def get_dataset_iter(split="train"):
    data_path = os.path.join(GPTConfig.data_dir, split)
    corpus_factory = get_corpus_iter(data_path)

    # Load tokenizer if exists, otherwise train and save
    try:
        tokenizer = BPETokenizer.load(GPTConfig.token_cache_dir)
        print("Tokenizer found. Loading tokenizer....")
    except Exception:
        print("No tokenizer found. Training tokenizer....")
        tokenizer = BPETokenizer()
        tokenizer.train(words_iter=corpus_factory())
        tokenizer.save(GPTConfig.token_cache_dir)

    print(f"Finished Loading/Training Tokenizer. Vocab size: {len(tokenizer)}. Creating dataset...")

    dataset = GPTIterableDataset(
        corpus_factory(),
        tokenizer,
        block_size=GPTConfig.block_size,
        stride=GPTConfig.stride,
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
