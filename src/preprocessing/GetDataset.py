import os
import sys
import re
import unicodedata
from torch.utils.data import DataLoader

# Ensure project root import works when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from BPETokenizer import BPETokenizer
from Dataset import GPTIterableDataset
from GPTConfig import GPTConfig

WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Unicode normalize + collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def iter_words_from_dir(data_dir: str):
    """
    Streams 'word-like' tokens and punctuation from all .txt files in data_dir.

    Important:
      - We do NOT yield whitespace tokens (regex excludes them).
      - If you want explicit word boundaries, you can insert "<SP>" between yielded tokens
        and include it in tokenizer specials/corpus (requires retraining).
    """
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = normalize_text(line)
                    if not line:
                        continue
                    for w in WORD_RE.findall(line):
                        yield w


def get_corpus_iter(data_path: str):
    """Factory so datasets can create a fresh iterator each time."""
    def factory():
        return iter_words_from_dir(data_path)
    return factory


def get_dataset_iter():
    corpus_factory = get_corpus_iter(GPTConfig.data_dir)

    # Load tokenizer if exists, otherwise train and save
    try:
        tokenizer = BPETokenizer.load(GPTConfig.cache_dir)
        print("Tokenizer found. Loading tokenizer....")
    except Exception:
        print("No tokenizer found. Training tokenizer....")
        tokenizer = BPETokenizer()
        tokenizer.train(words_iter=corpus_factory())
        tokenizer.save(GPTConfig.cache_dir)

    print(f"Finished Loading/Training Tokenizer. Vocab size: {len(tokenizer)}. Creating dataset...")

    dataset = GPTIterableDataset(
        corpus_factory(),
        tokenizer,
        block_size=GPTConfig.block_size,
        stride=GPTConfig.stride,
        add_space_between_words=getattr(GPTConfig, "add_space_between_words", True),
    )

    # DataLoader knobs (safe defaults if not present in GPTConfig)
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
