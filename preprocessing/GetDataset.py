import os
import sys
import re, unicodedata
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from BPETokenizer import BPETokenizer
from Utility import DATA_PATH, CACHE_PATH
from Dataset import GPTIterableDataset

WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def iter_words_from_dir(data_dir: str):
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".txt"): continue
            with open(os.path.join(root, fname), "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = normalize_text(line)
                    if not line: continue
                    # Split into "words" (includes punctuation as tokens)
                    for w in WORD_RE.findall(line):
                        yield w

def normalize_text(text):
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

def get_corpus_iter(data_path: str):
    def factory():
        return iter_words_from_dir(data_path)
    
    return factory

def get_dataset_iter():
    corpus_factory = get_corpus_iter(DATA_PATH)

    try:
        print('Tokenizer found. Loading tokenizer....')
        tokenizer = BPETokenizer.load(CACHE_PATH)
    except Exception:
        print('No tokenizer found. Training tokenizer....')
        tokenizer = BPETokenizer()
        tokenizer.train(words_iter=corpus_factory())
        tokenizer.save(CACHE_PATH)

    print(f'Finished Loading/Training Tokenizer. Vocab size: {len(tokenizer)}. Creating dataset...')
    dataset = GPTIterableDataset(corpus_factory(), tokenizer, block_size = 64, stride = 32)

    print('Created dataset. Creating DataLoader...')
    data_loader = DataLoader(dataset, batch_size = 8)
    return data_loader

data_loader_iter = iter(get_dataset_iter())
next_batch = next(data_loader_iter)
