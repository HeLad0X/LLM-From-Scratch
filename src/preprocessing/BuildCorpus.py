# build_corpus.py
# Streams text datasets and writes a fixed-size corpus to:
#   data/train/shard_XXXX.txt
#   data/eval/shard_XXXX.txt
#
# Token budget is approximated via chars-per-token until you train your tokenizer.
# Later you can re-count with your tokenizer if you want exact numbers.

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

from datasets import load_dataset


# --------------------------
# User settings
# --------------------------

OUT_ROOT = Path("data")     # will create data/train and data/eval
EVAL_RATIO = 0.01           # 1% eval is plenty at this scale
DOCS_PER_SHARD = 2000       # write a shard every N documents per split
MIN_CHARS = 300             # skip ultra-short snippets
CHARS_PER_TOKEN = 4.0       # heuristic: 1 token ~= 4 chars for English-ish BPE

# Total target tokens to write (estimate). Set whatever you want:
TOTAL_TARGET_TOKENS = 250_000_000  # example: 120M tokens

# How to divide across sources (must sum to 1.0)
# For "modern English" general GPT at your scale, this is a good start:
MIX = {
    "fineweb": 0.65,
    "wikipedia":   0.25,
    "hacker_news": 0.10,
    # "github_readme": 0.05,
}

# Dataset definitions (HF streaming)
# NOTE: These are the simplest modern-ish sources that don't require scraping.
SOURCES = {
        "fineweb": {
        "dataset": "HuggingFaceFW/fineweb",
        "subset": None,
        "split": "train",
        "text_key_priority": ["text"],
    },
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_key_priority": ["text"],
    },
    "hacker_news": {
        "dataset": "OpenPipe/hacker-news",
        "subset": None,
        "split": "train",
        "text_key_priority": ["text"],
    },
    # "github_readme": {
    #     "dataset": "codeparrot/github-readme-text",
    #     "subset": None,
    #     "split": "train",
    #     "text_key_priority": ["text"],
    # },
}


# --------------------------
# Helpers
# --------------------------

import unicodedata

def basic_clean(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\0", " ")
    s = " ".join(s.split())
    return s.strip()


def stable_split_bucket(text: str, eval_ratio: float) -> str:
    """
    Deterministic split based on hash of text:
    same text always goes to same split.
    """
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    r = int(h[:8], 16) / 0xFFFFFFFF
    return "eval" if r < eval_ratio else "train"

def extract_text(example: Dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # fallback: join all string fields
    parts = [v for v in example.values() if isinstance(v, str) and v.strip()]
    return "\n".join(parts) if parts else ""

import time
from datasets import load_dataset

def stream_text_resilient(dataset_name, subset=None, split="train",
                          text_key_priority=("text", "content"),
                          max_retries=20, backoff_s=5):
    """
    Streaming iterator that auto-recovers if HF/httpx throws 'client closed' or transient network errors.
    On recovery it restarts the stream (may re-yield a few docs; fine for corpus building).
    """
    attempt = 0
    while True:
        try:
            ds = load_dataset(dataset_name, subset, split=split, streaming=True)
            for ex in ds:
                # extract text
                text = ""
                for k in text_key_priority:
                    v = ex.get(k)
                    if isinstance(v, str) and v.strip():
                        text = v
                        break
                if not text:
                    parts = [v for v in ex.values() if isinstance(v, str) and v.strip()]
                    text = "\n".join(parts) if parts else ""
                if text:
                    yield text
            return  # exhausted cleanly
        except RuntimeError as e:
            msg = str(e).lower()
            if "client has been closed" not in msg and "cannot send a request" not in msg:
                raise
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(backoff_s * attempt)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(backoff_s * attempt)


def ensure_dirs():
    (OUT_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "eval").mkdir(parents=True, exist_ok=True)

class ShardWriter:
    def __init__(self, out_root: Path, docs_per_shard: int):
        self.out_root = out_root
        self.docs_per_shard = docs_per_shard
        self.buffers = {"train": [], "eval": []}
        self.shard_idx = {"train": 0, "eval": 0}
        self.doc_counts = {"train": 0, "eval": 0}
        self.char_counts = {"train": 0, "eval": 0}

    def _flush(self, split: str):
        if not self.buffers[split]:
            return
        path = self.out_root / split / f"shard_{self.shard_idx[split]:04d}.txt"
        path.write_text("\n".join(self.buffers[split]) + "\n", encoding="utf-8")
        self.buffers[split].clear()
        self.shard_idx[split] += 1

    def add(self, split: str, text: str):
        self.buffers[split].append(text)
        self.doc_counts[split] += 1
        self.char_counts[split] += len(text)

        if len(self.buffers[split]) >= self.docs_per_shard:
            self._flush(split)

    def close(self):
        self._flush("train")
        self._flush("eval")

def build_corpus():
    ensure_dirs()

    # Compute per-source char budgets from the token budget heuristic
    total_char_budget = int(TOTAL_TARGET_TOKENS * CHARS_PER_TOKEN)
    per_source_char_budget = {
        name: int(total_char_budget * frac)
        for name, frac in MIX.items()
    }

    writer = ShardWriter(OUT_ROOT, docs_per_shard=DOCS_PER_SHARD)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_target_tokens_est": TOTAL_TARGET_TOKENS,
        "chars_per_token_heuristic": CHARS_PER_TOKEN,
        "eval_ratio": EVAL_RATIO,
        "docs_per_shard": DOCS_PER_SHARD,
        "min_chars": MIN_CHARS,
        "mix": MIX,
        "sources": SOURCES,
        "per_source_char_budget": per_source_char_budget,
    }

    # Track how many chars we wrote per source
    written_chars_by_source = {k: 0 for k in MIX.keys()}
    written_docs_by_source = {k: 0 for k in MIX.keys()}

    print("Writing corpus to:", OUT_ROOT.resolve())
    print("Per-source char budgets (estimated from token budget):")
    for s, b in per_source_char_budget.items():
        print(f"  - {s:12s}: {b/1e6:.1f}M chars (~{b/CHARS_PER_TOKEN/1e6:.1f}M tokens est)")

    for source_name, frac in MIX.items():
        if source_name not in SOURCES:
            raise ValueError(f"Source '{source_name}' is in MIX but not defined in SOURCES.")

        spec = SOURCES[source_name]
        char_budget = per_source_char_budget[source_name]
 
        print(f"\n[Source: {source_name}] streaming {spec['dataset']} / {spec['subset']} ...")
        it = stream_text_resilient(
            dataset_name=spec["dataset"],
            subset=spec["subset"],
            split=spec["split"],
            text_key_priority=spec["text_key_priority"],
        )

        for raw in it:
            text = basic_clean(raw)
            if len(text) < MIN_CHARS:
                continue

            split = stable_split_bucket(text, EVAL_RATIO)
            writer.add(split, text)

            written_chars_by_source[source_name] += len(text)
            written_docs_by_source[source_name] += 1

            if written_chars_by_source[source_name] >= char_budget:
                break

        print(f"  -> wrote {written_chars_by_source[source_name]/1e6:.1f}M chars "
              f"(~{written_chars_by_source[source_name]/CHARS_PER_TOKEN/1e6:.1f}M tokens est), "
              f"docs={written_docs_by_source[source_name]:,}")

    writer.close()

    # Save corpus metadata
    meta["written_docs_by_split"] = writer.doc_counts
    meta["written_chars_by_split"] = writer.char_counts
    meta["written_chars_by_source"] = written_chars_by_source
    meta["written_docs_by_source"] = written_docs_by_source

    (OUT_ROOT / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total_chars = sum(writer.char_counts.values())
    print("\nDone.")
    print(f"Total written: {total_chars/1e6:.1f}M chars (~{total_chars/CHARS_PER_TOKEN/1e6:.1f}M tokens est)")
    print(f"Train chars: {writer.char_counts['train']/1e6:.1f}M | Eval chars: {writer.char_counts['eval']/1e6:.1f}M")
    print(f"Metadata: {(OUT_ROOT / 'corpus_meta.json').resolve()}")


if __name__ == "__main__":
    build_corpus()
