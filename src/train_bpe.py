import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

from GPTConfig import GPTConfig
from preprocessing.GetDataset import get_corpus_iter
from preprocessing.TokenizerFactory import load_or_build_tokenizer


def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_bpe_{ts}.log"

    logger = logging.getLogger("train_bpe")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_path}")
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train and cache the custom BPE tokenizer.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing tokenizer cache before training.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log every N cleaned documents while streaming corpus.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Load entire corpus into RAM before training (use only if corpus fits in memory).",
    )
    args = parser.parse_args()

    log_dir = Path(getattr(GPTConfig, "log_dir", "./logs"))
    logger = _setup_logger(log_dir)

    cache_dir = Path(getattr(GPTConfig, "token_cache_dir", "./cache_data/bpe_tokenizer"))
    cache_exists = cache_dir.exists()

    if cache_exists and not args.force:
        logger.info(f"Existing tokenizer cache found at {cache_dir}; loading without retraining.")
        tok = load_or_build_tokenizer()
        logger.info(f"Loaded cached tokenizer. Vocab size: {len(tok)}")
        print(f"BPE tokenizer ready. Vocab size: {len(tok)}. Cache: {cache_dir.resolve()}")
        return

    if args.force and cache_exists:
        logger.info(f"--force: removing existing cache at {cache_dir}")
        shutil.rmtree(cache_dir)

    # Prefer data/train if present, otherwise fall back to data root.
    data_root = Path(GPTConfig.data_dir or "data")
    train_dir = data_root / "train"
    corpus_path = train_dir if train_dir.is_dir() else data_root
    logger.info(f"Using corpus path: {corpus_path}")

    corpus_factory = get_corpus_iter(str(corpus_path))

    if args.in_memory:
        logger.info("Loading entire corpus into memory (use only if it fits).")
        cached_docs = []
        total_chars = 0
        for i, doc in enumerate(corpus_factory(), start=1):
            cached_docs.append(doc)
            total_chars += len(doc)
            if i % max(1, args.log_every) == 0:
                logger.info(f"Loaded {i} documents | approx {total_chars/1e6:.1f}M chars")
        logger.info(f"Finished loading corpus into memory: {len(cached_docs)} docs, ~{total_chars/1e6:.1f}M chars")

        def in_memory_iter():
            for doc in cached_docs:
                yield doc

        train_iter_factory = in_memory_iter
    else:
        def logging_corpus_iter():
            count = 0
            for doc in corpus_factory():
                count += 1
                if count % max(1, args.log_every) == 0:
                    logger.info(f"Cleaned documents emitted: {count}")
                yield doc
            logger.info(f"Corpus iteration finished. Total cleaned documents: {count}")

        train_iter_factory = logging_corpus_iter

    logger.info("Starting tokenizer load/train...")
    tok = load_or_build_tokenizer(train_iter_factory=train_iter_factory)

    logger.info(f"Tokenizer ready. Vocab size: {len(tok)}. Cache: {cache_dir.resolve()}")
    print(f"BPE tokenizer ready. Vocab size: {len(tok)}. Cache: {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
